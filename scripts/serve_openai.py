#!/usr/bin/env python3
"""OpenAI-compatible server for DeepSeek-V4.1-Flash on the PipeNetwork MLX port + my patch stack
+ the native DSpark (MTP) speculative decoder. Single-stream, greedy, with prefix-cache reuse.

    /tmp/dsv41/.venv/bin/python ~/qwen/v41/v41_server.py [--port 8001] [--max-seq 131072]

Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions (stream or not; tools; thinking).
Prompt format + tool-call parsing come from DeepSeek's own encoding.py (deepseek_v41_mlx/encoding.py).
Prefix cache: the KV cache of the previous request is kept; a new prompt sharing a prefix is
truncated back to the longest even common prefix (the compressor's open-group carry is derived
from start_pos % ratio, so even positions need no state reset) and only the tail is prefilled.
Env: V41_SPEC=0 disables DSpark; V41_THINKING=chat|thinking|auto (auto: thinking iff the request
carries reasoning_effort); V41_PREFILL_CHUNK (2048), V41_CACHE_SLOTS (4)."""
import argparse, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _cache_snapshot, _cache_restore
from deepseek_v41_mlx import dspark as D
from deepseek_v41_mlx import encoding as ENC

MODEL_DIR = None   # set from --model
MODEL_ID = "DeepSeek-V4.1-Flash"
EOS = 1
def log(*a): print(time.strftime("%H:%M:%S"), "[v41]", *a, file=sys.stderr, flush=True)


class Engine:
    def __init__(self, max_seq):
        t0 = time.time()
        m = load(MODEL_DIR); self.model = m[0] if isinstance(m, tuple) else m
        self.tok = load_tokenizer(MODEL_DIR)
        self.spec = os.environ.get("V41_SPEC", "1") == "1"
        self.spec = self.spec and os.path.exists(os.path.join(MODEL_DIR, "dspark", "dspark.safetensors"))
        self.drafter = D.load_dspark(self.model, MODEL_DIR) if self.spec else None
        self.model._dspark_targets = self.drafter.targets if self.spec else [37, 38, 39]
        self.max_seq = max_seq
        self.cache = None; self.covered = []; self.slot = None; self.slots = []
        self.lock = threading.Lock()
        try: mx.set_wired_limit(int(470e9))
        except Exception: pass
        log(f"loaded in {time.time()-t0:.0f}s, peak {mx.get_peak_memory()/1e9:.0f} GB, dspark={self.spec}")

    # ---- prefix cache: a few slots (pi interleaves 20k-token agent turns with tiny title/summary requests) ----
    SLOTS = int(os.environ.get("V41_CACHE_SLOTS", "4"))

    def _new_slot(self):
        cache = self.model.make_cache(bsz=1, max_seq_len=self.max_seq, dtype=mx.bfloat16)
        rings = [mx.zeros((1, self.drafter.window, self.model.args.head_dim), dtype=mx.bfloat16) for _ in self.drafter.stages] if self.spec else None
        return {"cache": cache, "covered": [], "rings": rings, "last_pos": -1, "used": 0.0}

    def _select(self, ids):
        """Pick the slot with the longest common prefix (>= 64 tokens), else the LRU slot, reset."""
        if not hasattr(self, "slots"): self.slots = []
        best, best_n = None, 0
        for sl in self.slots:
            cov = sl["covered"]; n = 0; lim = min(len(ids), len(cov))
            while n < lim and ids[n] == cov[n]: n += 1
            if n > best_n: best, best_n = sl, n
        if best is None or best_n < 64:
            if len(self.slots) < self.SLOTS: best = self._new_slot(); self.slots.append(best)
            else:
                best = min(self.slots, key=lambda x: x["used"])
                best["cache"].offset = 0; best["covered"] = []; best["last_pos"] = -1
            best_n = 0
        best["used"] = time.time()
        return best, best_n

    def prepare(self, ids):
        """Bring a cache slot to hold all of ids; returns the logits of the last position."""
        if len(ids) + 4096 > self.max_seq: raise ValueError(f"prompt of {len(ids)} tokens exceeds max_seq {self.max_seq}")
        sl, n = self._select(ids)
        T = (n // 2) * 2                                                                   # even: compressor carry is empty
        if T > len(ids) - 1: T = ((len(ids) - 1) // 2) * 2                                 # always prefill >= 1 token
        self.cache = sl["cache"]; self.cache.offset = T; self.covered = ids[:T]; self.slot = sl
        if self.spec:
            self.drafter.rings = sl["rings"]; self.drafter.last_pos = T - 1
        t0 = time.perf_counter(); logits = None; chunk = int(os.environ.get("V41_PREFILL_CHUNK", "2048"))
        for a in range(T, len(ids), chunk):
            piece = mx.array([ids[a:a + chunk]])
            logits, mh = D.forward_capture(self.model, piece, self.cache)
            mx.eval(logits)
            if self.spec: self.drafter.seed(mh, a)
            self.covered = ids[:a + piece.shape[1]]
        dt = time.perf_counter() - t0
        log(f"prefill: {len(ids)} prompt tokens, {T} reused, {len(ids)-T} computed in {dt:.2f}s ({(len(ids)-T)/max(dt,1e-6):.0f} tok/s) [slot {self.slots.index(sl)}/{len(self.slots)}]")
        return logits

    def _sync_slot(self):
        self.slot["covered"] = self.covered
        if self.spec: self.slot["rings"] = self.drafter.rings; self.slot["last_pos"] = self.drafter.last_pos

    def generate(self, ids, max_new, stop_ids=(EOS,)):
        """Yields committed token ids. Greedy; DSpark speculative if enabled."""
        try:
            yield from self._generate(ids, max_new, stop_ids)
        finally:
            self._sync_slot()

    def _generate(self, ids, max_new, stop_ids):
        logits = self.prepare(ids)
        first = int(mx.argmax(logits[:, -1], axis=-1)[0])
        seq = list(ids); out = []
        if not self.spec:
            tok = first
            while len(out) < max_new:
                out.append(tok); yield tok
                if tok in stop_ids: return
                lg = self.model(mx.array([[tok]]), self.cache, last_logit_only=True)
                tok = int(mx.argmax(lg[0, -1])); self.covered = seq + out
            return
        dr = self.drafter; B = dr.block_size; pending = [first]; out = [first]
        yield first
        if first in stop_ids: return
        st = D.STATS
        for k in ("steps", "drafted", "accepted", "rejects", "resyncs"): st[k] = 0
        while len(out) < max_new:
            p0 = self.cache.offset
            if D.CONF_MIN is None:
                draft = dr.draft(pending[-1])
            else:                                            # S2 confidence trimming (see dspark.CONF_MIN)
                _, draft, conf = dr.draft_logits(pending[-1], with_confidence=True)
                keep = 0
                for c in conf[0].tolist():
                    if c < D.CONF_MIN: break
                    keep += 1
                draft = draft[:max(keep, 1)]
            inp = pending + draft
            st["steps"] += 1; st["drafted"] += len(draft)
            snap = _cache_snapshot(self.cache)
            lg, mh = D.forward_capture(self.model, mx.array([inp]), self.cache)
            preds = mx.argmax(lg[0], axis=-1).tolist()
            P = len(pending); acc = 0
            for j, d in enumerate(draft):
                if preds[P - 1 + j] == d: acc += 1
                else: break
            st["accepted"] += acc
            new = draft[:acc] + [preds[P - 1 + acc]]
            dr.seed(mh[:, :P + acc], p0)
            if acc == len(draft):
                pending = [new[-1]]
            else:
                st["rejects"] += 1; _cache_restore(self.cache, snap)
                pending = pending + new
                if len(pending) > 12:
                    st["resyncs"] += 1; D.forward_capture(self.model, mx.array([pending[:-1]]), self.cache); pending = [new[-1]]
            self.covered = (seq + out + new)[:self.cache.offset]
            for t in new:
                out.append(t); yield t
                if t in stop_ids or len(out) >= max_new: return


ENGINE = None


def to_text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


def build_messages(req):
    msgs = []
    for m in req.get("messages", []):
        m = dict(m); m["content"] = to_text(m.get("content"))
        if m.get("role") == "assistant" and not m.get("content") and not m.get("tool_calls"): m["content"] = ""
        msgs.append(m)
    tools = req.get("tools")
    if tools:
        if msgs and msgs[0].get("role") == "system": msgs[0]["tools"] = tools
        else: msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return msgs


def thinking_mode(req):
    mode = os.environ.get("V41_THINKING", "auto")
    if mode in ("chat", "thinking"): return mode
    return "thinking" if req.get("reasoning_effort") else "chat"


def effort(req):
    e = req.get("reasoning_effort")
    if e is None: return None
    return {"low": 50, "medium": 60, "high": 75, "max": 100}.get(e, e) if isinstance(e, str) else int(e)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        elif self.path.startswith("/health"):
            self._json(200, {"status": "ok", "model": MODEL_ID, "dspark": ENGINE.spec, "cache_slots": [len(s["covered"]) for s in ENGINE.slots]})
        else: self._json(404, {"error": "not found"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.startswith("/v1/chat/completions"): return self._json(404, {"error": "not found"})
        try: self.chat(req)
        except BrokenPipeError: log("client disconnected")
        except Exception as e:
            import traceback; log("error:", repr(e)); log(traceback.format_exc())
            try: self._json(500, {"error": {"message": str(e)}})
            except Exception: pass
    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush()
    def chat(self, req):
        mode = thinking_mode(req); msgs = build_messages(req)
        prompt = ENC.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort(req))
        ids = ENGINE.tok(prompt, add_special_tokens=False)["input_ids"]
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
        stops = req.get("stop") or []; stops = [stops] if isinstance(stops, str) else list(stops)
        stream = bool(req.get("stream")); rid = "chatcmpl-" + uuid.uuid4().hex[:24]; created = int(time.time())
        tc_marker = f"\n\n<{ENC.dsml_token}{ENC.tool_calls_block_name}"
        HOLD = len(tc_marker) + 4
        if stream:
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def sse(delta, finish=None, usage=None):
            ch = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage: ch["usage"] = usage
            self._chunk(f"data: {json.dumps(ch)}\n\n".encode())
        with ENGINE.lock:
            t0 = time.perf_counter(); out = []; text = ""; sent = 0; phase = "reasoning" if mode == "thinking" else "content"
            if stream: sse({"role": "assistant", "content": ""})
            finish = "stop"; tdec = None
            for tok in ENGINE.generate(ids, max_new):
                if tdec is None: tdec = time.perf_counter()
                if tok == EOS: break
                out.append(tok)
                text = ENGINE.tok.decode(out, skip_special_tokens=False)
                if any(s in text for s in stops): finish = "stop"; break
                if not stream or text.endswith("�"): continue
                # stream reasoning / content deltas, holding back a tail that could be a tool-call marker
                if phase == "reasoning":
                    k = text.find(ENC.thinking_end_token)
                    if k >= 0:
                        if k > sent: sse({"reasoning_content": text[sent:k]})
                        sent = k + len(ENC.thinking_end_token); phase = "content"
                    else:
                        cut = len(text) - len(ENC.thinking_end_token)
                        if cut > sent: sse({"reasoning_content": text[sent:cut]}); sent = cut
                if phase == "content":
                    k = text.find(tc_marker, sent)
                    if k >= 0:
                        if k > sent: sse({"content": text[sent:k]})
                        sent = k; phase = "tools"
                    else:
                        cut = len(text) - HOLD
                        if cut > sent: sse({"content": text[sent:cut]}); sent = cut
            else:
                finish = "length"
            if len(out) >= max_new: finish = "length"
            dt = time.perf_counter() - t0; ddt = time.perf_counter() - (tdec or t0)
            # final parse (DeepSeek's own parser); fall back to raw text on malformed output
            parsed = None
            try: parsed = ENC.parse_message_from_completion_text(text + ENC.eos_token, thinking_mode=mode)
            except Exception as e:
                log("parse fallback:", str(e)[:80])
                rc, ct = "", text
                if mode == "thinking" and ENC.thinking_end_token in text:
                    rc, ct = text.split(ENC.thinking_end_token, 1)
                parsed = {"content": ct, "reasoning_content": rc, "tool_calls": []}
            tool_calls = [{"id": tc.get("id") or "call_" + uuid.uuid4().hex[:12], "type": "function",
                           "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]
                                        if isinstance(tc["function"]["arguments"], str) else json.dumps(tc["function"]["arguments"])}}
                          for tc in parsed.get("tool_calls") or []]
            if tool_calls: finish = "tool_calls"
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
            st = D.STATS if ENGINE.spec else {}
            log(f"gen: {len(out)} tokens in {ddt:.1f}s ({len(out)/max(ddt,1e-6):.1f} tok/s decode, {dt:.1f}s total)"
                + (f", dspark {st.get('accepted',0)}/{st.get('drafted',0)} accepted, {len(out)/max(st.get('steps',1),1):.2f} tok/step" if ENGINE.spec else "")
                + f", mode={mode}, finish={finish}, tool_calls={len(tool_calls)}")
        if stream:
            # flush whatever content/reasoning was held back, then tool calls
            if phase == "reasoning" and parsed.get("reasoning_content") and len(parsed["reasoning_content"]) > sent:
                sse({"reasoning_content": parsed["reasoning_content"][sent:]})
            elif phase == "content":                # content deltas were sliced from `text`: emit the held-back remainder up to any tool marker
                k = text.find(tc_marker, sent)
                tail = text[sent:k] if k >= 0 else text[sent:]
                if tail: sse({"content": tail})
            if tool_calls:
                sse({"tool_calls": [{"index": i, "id": tc["id"], "type": "function", "function": tc["function"]} for i, tc in enumerate(tool_calls)]})
            sse({}, finish, usage)
            self._chunk(b"data: [DONE]\n\n"); self._chunk(b"")
        else:
            msg = {"role": "assistant", "content": parsed.get("content") or ""}
            if parsed.get("reasoning_content"): msg["reasoning_content"] = parsed["reasoning_content"]
            if tool_calls: msg["tool_calls"] = tool_calls
            self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="converted MLX model dir (with <model>/dspark from scripts/convert_mtp.py for speculative decoding)")
    ap.add_argument("--port", type=int, default=8001); ap.add_argument("--max-seq", type=int, default=131072)
    a = ap.parse_args(); MODEL_DIR = a.model
    ENGINE = Engine(a.max_seq)
    # single-threaded on purpose: MLX streams are bound to the thread that first used them
    # ("There is no Stream(gpu, N) in current thread" from handler threads); requests queue at the socket
    srv = HTTPServer(("127.0.0.1", a.port), Handler); srv.request_queue_size = 64
    log(f"serving {MODEL_ID} on http://127.0.0.1:{a.port}/v1")
    srv.serve_forever()
