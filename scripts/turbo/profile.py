"""Per-component decode profile. Patches __call__ at the CLASS level (instance-level
wrappers are ignored: Python resolves __call__ on the type). Forced eval after each
component => read SHARES, the absolute tok/s is inflated by the syncs."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time, collections
import mlx.core as mx

if os.environ.get("V41_PATCH"):
    from deepseek_v41_mlx.turbo import patches as v41_patches; v41_patches.apply(os.environ["V41_PATCH"].split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer
from deepseek_v41_mlx import attention as A, moe as MOE, engram as E, layers as L, compressor as CP, indexer as IX, sparse_attention as SA, fakequant as FQ, hyper_connections as HC
T = collections.defaultdict(float); C = collections.Counter()
def wrap(cls, name):
    orig = cls.__call__
    def w(self, *a, **k):
        t = time.perf_counter(); out = orig(self, *a, **k)
        mx.eval(out) if isinstance(out, mx.array) else mx.eval([o for o in (out if isinstance(out, tuple) else ()) if isinstance(o, mx.array)])
        T[name] += time.perf_counter() - t; C[name] += 1; return out
    cls.__call__ = w
def wrapfn(mod, fname, name):
    orig = getattr(mod, fname)
    def w(*a, **k):
        t = time.perf_counter(); out = orig(*a, **k)
        mx.eval(out) if isinstance(out, mx.array) else mx.eval(list(out))
        T[name] += time.perf_counter() - t; C[name] += 1; return out
    setattr(mod, fname, w)
wrap(A.Attention, "attn (whole)"); wrap(MOE.MoE, "ffn/moe (whole)"); wrap(E.Engram, "engram (whole)")
wrap(L.RMSNorm, "  rmsnorm"); wrap(CP.Compressor, "  compressor"); wrap(IX.Indexer, "  indexer")
for cls in (E.EngramEmbedding, E.QuantizedEngramEmbedding): wrap(cls, "  engram.gather+dequant")
wrapfn(SA, "sparse_attn", "  sparse_attn"); wrapfn(L, "rope_tail", "  rope_tail")
for f in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_ue8m0", "fake_quant_fp4_e4m3"): wrapfn(FQ, f, "  fakequant")
wrapfn(A, "window_idx_matrix", "  window_idx_matrix"); wrapfn(HC, "hc_mixes", "  hc_mixes"); wrapfn(HC, "hc_pre", "  hc_pre"); wrapfn(HC, "hc_post", "  hc_post")
# rope_tail/fakequant/window_idx are called INSIDE attn; hc_* inside Block -> shares overlap by design
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit"); N = int(os.environ.get("N_DECODE", "12")); TOKN = int(os.environ.get("TOKN", "1"))
t0 = time.time(); model = load(M); model = model[0] if isinstance(model, tuple) else model; tok = load_tokenizer(M)
print(f"[prof] load {time.time()-t0:.1f}s", flush=True)
hasher = model.engram_hasher
if hasher is not None:
    ho = type(hasher).__call__
    def hw(self, *a, **k):
        t = time.perf_counter(); r = ho(self, *a, **k); T["  hasher(numpy,cpu)"] += time.perf_counter()-t; C["  hasher(numpy,cpu)"] += 1; return r
    type(hasher).__call__ = hw
ids = tok("Explain how a copy-on-write B-tree keeps a reader consistent during a page split.")["input_ids"]
if os.environ.get("POST"): v41_patches.apply_post_load(model, os.environ["POST"].split(","))
cache = model.make_cache(bsz=1, max_seq_len=len(ids)+N*TOKN+8, dtype=mx.bfloat16)
logits = model(mx.array([ids]), cache, last_logit_only=True); mx.eval(logits); T.clear(); C.clear()
tk = mx.argmax(logits[:, -1], axis=-1); per = []
for i in range(N):
    t = time.perf_counter(); inp = tk[:, None] if TOKN == 1 else mx.array([ids[:TOKN]]); logits = model(inp, cache, last_logit_only=True); mx.eval(logits)
    tk = mx.argmax(logits[:, -1], axis=-1); mx.eval(tk); per.append(time.perf_counter()-t)
tot = sum(per)
print(f"[prof] TOKN={TOKN} {N} steps: mean {tot/N*1000:.0f} ms/tok (instrumented; min {min(per)*1000:.0f} max {max(per)*1000:.0f})\n")
print(f"  {'component':<28}{'ms/tok':>9}{'calls/tok':>11}{'ms/call':>9}{'share':>8}"); print("  "+"-"*67)
for k, v in sorted(T.items(), key=lambda x: (x[0].startswith("  "), -x[1])):
    print(f"  {k:<28}{v/N*1000:>9.1f}{C[k]/N:>11.0f}{v*1000/max(C[k],1):>9.3f}{v/tot*100:>7.1f}%")
top = sum(v for k, v in T.items() if not k.startswith("  "))
print(f"  {'(glue outside attn/ffn/engram)':<28}{(tot-top)/N*1000:>9.1f}{'':>11}{'':>9}{(tot-top)/tot*100:>7.1f}%")
