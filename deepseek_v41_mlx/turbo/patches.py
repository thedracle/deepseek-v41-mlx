"""Opt-in runtime patches for the PipeNetwork DeepSeek-V4.1 port, applied by monkeypatch
so the port itself stays untouched and each patch can be A/B'd independently.

    from deepseek_v41_mlx.turbo import patches as v41_patches; v41_patches.apply(["sinkhorn"])
"""
import sys
import mlx.core as mx

def patch_sinkhorn():
    """split_sinkhorn runs 20 Sinkhorn sweeps in a Python loop on a tiny tensor, and is
    called twice per layer (80x per decoded token) -> ~1,600 kernel launches/token.
    mx.compile fuses the loop. Micro-bench: 0.921 -> 0.568 ms/call, max abs diff 1.5e-8."""
    from deepseek_v41_mlx import hyper_connections as hc
    orig = hc.split_sinkhorn
    _cache = {}
    def compiled_split(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        key = (hc_mult, sinkhorn_iters, eps)
        fn = _cache.get(key)
        if fn is None:
            fn = _cache[key] = mx.compile(
                lambda m, s, b: orig(m, s, b, hc_mult, sinkhorn_iters, eps))
        return fn(mixes, hc_scale, hc_base)
    hc.split_sinkhorn = compiled_split
    # hc_mixes captured the name at import; rebind through the module global it reads
    return "sinkhorn: split_sinkhorn -> mx.compile"

def patch_nosync():
    """greedy_generate does `int(tok[0])` every token: a GPU->CPU round-trip that drains
    the pipeline before the next forward can be enqueued. Keep the token on-device and
    only materialise the ids at the end (EOS check batched every 16 tokens)."""
    from deepseek_v41_mlx import generate as G
    from deepseek_v41_mlx.generate import _forward
    def greedy_generate(model, input_ids, max_new_tokens=64, max_seq_len=None,
                        eos_id=1, dtype=mx.float32, prefill_chunk=512, eos_check_every=16):
        try: mx.set_wired_limit(int(470e9))
        except Exception: pass
        ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
        total = ids.shape[1] + max_new_tokens
        cache = model.make_cache(bsz=ids.shape[0], max_seq_len=max_seq_len or (total + 8), dtype=dtype)
        logits = None
        for a in range(0, ids.shape[1], prefill_chunk):
            logits = _forward(model, ids[:, a:a + prefill_chunk], cache)
        toks = []
        tok = mx.argmax(logits[:, -1], axis=-1)
        for i in range(max_new_tokens):
            toks.append(tok)                          # stays on device, no sync
            logits = _forward(model, tok[:, None], cache)
            tok = mx.argmax(logits[:, -1], axis=-1)
            if eos_id >= 0 and (i + 1) % eos_check_every == 0:
                recent = mx.stack(toks[-eos_check_every:]); mx.eval(recent)
                hit = [j for j, v in enumerate(recent[:, 0].tolist()) if v == eos_id]
                if hit:
                    toks = toks[:len(toks) - eos_check_every + hit[0]]; break
        out = mx.stack(toks)[:, 0].tolist() if toks else []
        if eos_id >= 0 and eos_id in out: out = out[:out.index(eos_id)]
        return out
    G.greedy_generate = greedy_generate
    return "nosync: greedy_generate keeps tokens on-device (no per-token int() sync)"

def patch_sinkhorn_metal():
    """Native Metal kernel: the entire split_sinkhorn (pre/post sigmoids + 20 Sinkhorn
    sweeps on the 4x4 comb) in registers, one thread per token, ONE launch instead of
    ~40. Verified vs eager: max|diff| <= 1.2e-7. Micro-bench 1.298 -> 0.283 ms/call."""
    from deepseek_v41_mlx import hyper_connections as hc
    from deepseek_v41_mlx.turbo import metal as v41_metal
    hc.split_sinkhorn = v41_metal.sinkhorn_split
    return "sinkhorn_metal: split_sinkhorn -> native Metal kernel (1 launch)"

def patch_hc_metal():
    """Fused Metal kernels for hc_post (out[k,d]=post[k]*x[d]+sum_j comb[j,k]*res[j,d]) and
    hc_pre (collapse). One launch each, no [b,s,4,4,d] intermediate. Verified: fp32 max|diff|
    4.8e-7; bf16 hc_post differs by one bf16 ulp from summation order. 160 calls/token."""
    from deepseek_v41_mlx import hyper_connections as hc, model as M
    from deepseek_v41_mlx.turbo import metal as v41_metal
    hc.hc_post, hc.hc_pre = v41_metal.hc_post, v41_metal.hc_pre
    # model.py may have imported the names directly; rebind those too if present
    for name in ("hc_post", "hc_pre"):
        if hasattr(M, name): setattr(M, name, getattr(v41_metal, name))
    return "hc_metal: hc_post/hc_pre -> native Metal kernels"

def patch_rmsnorm():
    """Port RMSNorm = 4 launches (astype, mean(sq), rsqrt*mul, weight*astype). mx.fast.rms_norm
    is 1. With fp32 input and cast-out it is BIT-EXACT vs the port (verified D=5120/1280/512).
    160 calls/token -> ~480 launches saved."""
    from deepseek_v41_mlx import layers as L
    def call(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)
    L.RMSNorm.__call__ = call
    return "rmsnorm: RMSNorm -> mx.fast.rms_norm (bit-exact)"

def patch_idxcache():
    """All 40 layers recompute IDENTICAL window index math per token (window_idx_matrix,
    window_chrono slots, write_window slots depend only on pos/n/window). Memoise on the
    int key so it is computed once per forward, ~360 launches/token saved. Pure Python."""
    from deepseek_v41_mlx import attention as A, cache as CC
    _m = {}
    orig_wim = A.window_idx_matrix
    def wim(wp, n, window):
        k = ("wim", wp, n, window); v = _m.get(k)
        if v is None: v = _m[k] = orig_wim(wp, n, window)
        return v
    A.window_idx_matrix = wim
    orig_chrono = CC.LayerCache.window_chrono
    def chrono(self, pos):
        w = self.window; wp = min(pos, w)
        if wp == 0: return self.win_kv[:, :0]
        k = ("chrono", pos, w); slots = _m.get(k)
        if slots is None: slots = _m[k] = (pos - wp + mx.arange(wp)) % w
        return self.win_kv[:, slots]
    CC.LayerCache.window_chrono = chrono
    orig_write = CC.LayerCache.write_window
    def write(self, pos, kv):
        n = kv.shape[1]; keep = min(n, self.window)
        k = ("write", pos, n, self.window); slots = _m.get(k)
        if slots is None: slots = _m[k] = (pos + n - keep + mx.arange(keep)) % self.window
        self.win_kv[:, slots] = kv[:, n - keep:].astype(self.dtype)
    CC.LayerCache.write_window = write
    return "idxcache: window index math memoised across the 40 layers"

def patch_fq_metal():
    """Fake-quant chains -> single Metal kernels. fp8_ue8m0 verified BIT-EXACT vs the port
    incl. e4m3 rounding ties and subnormals (0 mismatched elements). ~25 launches -> 1 per call.
    fp4 variants are registered only if v41_metal provides them (verified separately)."""
    from deepseek_v41_mlx import fakequant as FQ, attention as A, indexer as IX
    from deepseek_v41_mlx.turbo import metal as v41_metal
    names = []
    for f in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_ue8m0", "fake_quant_fp4_e4m3"):
        k = getattr(v41_metal, f, None)
        if k is None: continue
        setattr(FQ, f, k)
        for mod in (A, IX):                       # they imported the names directly
            if hasattr(mod, f): setattr(mod, f, k)
        names.append(f.replace("fake_quant_", ""))
    return f"fq_metal: {','.join(names)} -> native Metal kernels"

def patch_rope_metal():
    """rope_tail (adjacent-pair rotation on the last rd channels) -> one Metal kernel instead of
    ~12 launches. ~136 calls/token. Verified vs layers.rope_tail on 3D/4D, forward and inverse."""
    from deepseek_v41_mlx import layers as L, attention as A, indexer as IX
    from deepseek_v41_mlx.turbo import metal as v41_metal
    L.rope_tail = v41_metal.rope_tail
    for mod in (A, IX):
        if hasattr(mod, "rope_tail"): mod.rope_tail = v41_metal.rope_tail
    return "rope_metal: rope_tail -> native Metal kernel"

def patch_sa_metal():
    """OPT-IN, NOT in the default stack. Fused sparse attention (gather + masked softmax w/ sink +
    weighted sum). Numerically correct (1 bf16 ulp vs port; exact on all-masked) but the v1
    kernel is SLOWER than the port's einsum path (naive per-thread gathers, serial reductions).
    Needs a tiled/simdgroup rewrite before it pays off. Kept for further work."""
    from deepseek_v41_mlx import sparse_attention as SA, attention as A
    from deepseek_v41_mlx.turbo import metal as v41_metal
    SA.sparse_attn = v41_metal.sparse_attn3      # v3 (split-K): best of three, 0.711 vs port 0.401 ms
    if hasattr(A, "sparse_attn"): A.sparse_attn = v41_metal.sparse_attn3
    return "sa_metal: sparse_attn -> fused Metal kernel v3 (correct, still 1.8x slower; opt-in)"

def patch_hcmix_metal():
    """OPT-IN, SLOWER (7x). Fully fused hc_mixes (rms + 24x20480 GEMV + sinkhorn) in one
    32-thread threadgroup. Numerically fp32-ulp exact, but a single-threadgroup GEMV loses badly
    to MLX's matmul. Lesson recorded: fuse elementwise/reduction chains, leave GEMVs to MLX."""
    from deepseek_v41_mlx import hyper_connections as hc, model as M
    from deepseek_v41_mlx.turbo import metal as v41_metal
    hc.hc_mixes = v41_metal.hc_mixes
    if hasattr(M, "hc_mixes"): M.hc_mixes = v41_metal.hc_mixes
    return "hcmix_metal: hc_mixes -> fully fused kernel (v1, slower; opt-in)"

def patch_gate_metal():
    """OPT-IN. Fused MoE gate (384 dots + sqrtsoftplus + bias + top-6 + normalise). Expert sets
    and weights match the port (2.7e-7). Same single-threadgroup-GEMV flaw as hcmix; see timing."""
    from deepseek_v41_mlx import moe as MOE
    from deepseek_v41_mlx.turbo import metal as v41_metal
    MOE.Gate.__call__ = v41_metal.moe_gate
    return "gate_metal: MoE Gate -> fused kernel (v1; opt-in)"

def patch_sa_sdpa():
    """Sparse attention via mx.fast.scaled_dot_product_attention (MLX's tuned flash-decode
    kernel). The attention sink becomes ONE extra key with a zero value vector whose logit is
    set to attn_sink[h] through the additive mask -> identical to the port's denominator-only
    sink. Masked idx -> -1e30 in the mask. MLA (1 KV head / 64 q heads) is SDPA's GQA fast path.
    Verified vs port: max diff 1.2e-4 (bf16 ulp), exact on all-masked. 0.461 vs 0.510 ms and
    ~20 -> ~7 launches at m==1. m>1 (speculative verify chunks, prefill): every query has its own
    key set, so each query becomes its own SDPA batch row: Q [b*m,h,1,d] vs its gathered
    K=V [b*m,1,k+1,d] -- one kernel instead of the port's einsum/where/exp/sum chain."""
    from deepseek_v41_mlx import sparse_attention as SA, attention as A
    import os
    from deepseek_v41_mlx.turbo import metal as v41_metal
    _SA_PREFILL = os.environ.get("V41_SA_PREFILL", "port")   # v4 loses in situ (330 vs 365 tok/s): real index lists are localized, the port's einsum is a batched GEMM
    orig = SA.sparse_attn
    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
        b, m, h, d = q.shape
        n = kv.shape[1]; k = topk_idxs.shape[-1]
        if m > 1 and m * k > 4096:          # prefill / big verify chunks: per-row SDPA loses past ~4k gathered keys
            if _SA_PREFILL == "v4" and d == 512 and h % 32 == 0:
                return v41_metal.sparse_attn4(q, kv, attn_sink, topk_idxs, softmax_scale)   # tiled Metal kernel, 1.2-2.9x vs port
            return orig(q, kv, attn_sink, topk_idxs, softmax_scale, chunk)
        safe = mx.maximum(topk_idxs, 0).astype(mx.int32)
        base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
        g = kv.reshape(b * n, d)[(safe + base).reshape(-1)].reshape(b * m, k, d)
        kvx = mx.concatenate([g, mx.zeros((b * m, 1, d), dtype=g.dtype)], axis=1)[:, None].astype(mx.float32)  # [b*m,1,k+1,d]
        Q = q.reshape(b * m, h, 1, d).astype(mx.float32)
        mask = mx.where(topk_idxs.reshape(b * m, 1, 1, k) >= 0, 0.0, -1e30).astype(mx.float32)
        mask = mx.concatenate([mx.broadcast_to(mask, (b * m, h, 1, k)),
                               mx.broadcast_to(attn_sink.astype(mx.float32).reshape(1, h, 1, 1), (b * m, h, 1, 1))], axis=-1)
        o = mx.fast.scaled_dot_product_attention(Q, kvx, kvx, scale=softmax_scale, mask=mask)
        return o.reshape(b, m, h, d).astype(q.dtype)
    SA.sparse_attn = sparse_attn
    if hasattr(A, "sparse_attn"): A.sparse_attn = sparse_attn
    return "sa_sdpa: sparse_attn -> mx.fast SDPA with sink-as-zero-value-key (any m; per-query batch rows)"

def patch_sa_sdpa_bf16():
    """OPT-IN. Same as sa_sdpa but runs SDPA in bf16: 0.323 vs 0.353 ms (fp32) at a numerics cost
    (max diff vs port 4.9e-4 instead of 1.2e-4). The port computes attention in fp32; keep fp32
    unless you have checked the model's outputs at bf16."""
    from deepseek_v41_mlx import sparse_attention as SA, attention as A
    orig = SA.sparse_attn
    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
        b, m, h, d = q.shape
        if m != 1: return orig(q, kv, attn_sink, topk_idxs, softmax_scale, chunk)
        n = kv.shape[1]; k = topk_idxs.shape[-1]
        g = kv.reshape(b*n, d)[(mx.maximum(topk_idxs, 0).astype(mx.int32) + (mx.arange(b, dtype=mx.int32)*n).reshape(b,1,1)).reshape(-1)].reshape(b, k, d)
        kvx = mx.concatenate([g, mx.zeros((b, 1, d), dtype=g.dtype)], axis=1)[:, None]
        mask = mx.concatenate([mx.broadcast_to(mx.where(topk_idxs.reshape(b,1,1,k) >= 0, 0.0, -1e30).astype(mx.bfloat16), (b,h,1,k)),
                               mx.broadcast_to(attn_sink.astype(mx.bfloat16).reshape(1,h,1,1), (b,h,1,1))], axis=-1)
        return mx.fast.scaled_dot_product_attention(q.reshape(b,h,1,d), kvx, kvx, scale=softmax_scale, mask=mask).reshape(b,1,h,d).astype(q.dtype)
    SA.sparse_attn = sparse_attn
    if hasattr(A, "sparse_attn"): A.sparse_attn = sparse_attn
    return "sa_sdpa_bf16: SDPA sparse attention in bf16 (opt-in, small numerics cost)"

def patch_async():
    """Pipeline the decode loop. The port's _forward does mx.eval(lg) every step, so the CPU builds
    step N+1's graph (~3,000 ops of Python dispatch) only AFTER the GPU finishes step N. mlx-lm
    overlaps them with mx.async_eval: enqueue step N, immediately build and enqueue N+1, and only
    block when a value is actually needed. Drops the per-step Metal-timeout snapshot/CPU-fallback
    at decode (kept for prefill chunks) — acceptable for benchmarking; documented."""
    from deepseek_v41_mlx import generate as G
    orig_forward = G._forward
    def greedy_generate(model, input_ids, max_new_tokens=64, max_seq_len=None, eos_id=1,
                        dtype=mx.float32, prefill_chunk=512, eos_check_every=16):
        try: mx.set_wired_limit(int(470e9))
        except Exception: pass
        ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
        total = ids.shape[1] + max_new_tokens
        cache = model.make_cache(bsz=ids.shape[0], max_seq_len=max_seq_len or (total + 8), dtype=dtype)
        logits = None
        for a in range(0, ids.shape[1], prefill_chunk):
            logits = orig_forward(model, ids[:, a:a + prefill_chunk], cache)   # prefill keeps the safe path
        toks = []
        tok = mx.argmax(logits[:, -1], axis=-1)
        mx.async_eval(tok)
        for i in range(max_new_tokens):
            toks.append(tok)
            logits = model(tok[:, None], cache, last_logit_only=True)          # no eval here
            tok = mx.argmax(logits[:, -1], axis=-1)
            mx.async_eval(tok)                                                  # enqueue; do not block
            if eos_id >= 0 and (i + 1) % eos_check_every == 0:
                recent = mx.stack(toks[-eos_check_every:]); mx.eval(recent)
                hit = [j for j, v in enumerate(recent[:, 0].tolist()) if v == eos_id]
                if hit: toks = toks[:len(toks) - eos_check_every + hit[0]]; break
        out = mx.stack(toks)[:, 0].tolist() if toks else []
        if eos_id >= 0 and eos_id in out: out = out[:out.index(eos_id)]
        return out
    G.greedy_generate = greedy_generate
    return "async: decode loop pipelined with mx.async_eval (graph build overlaps GPU)"

def patch_wo_a_f32():
    """wo_a is stored unquantized bf16 (block-diagonal, cannot be MLX-quantized). The port does
    `self.wo_a.weight.reshape(g, r, -1).astype(float32)` inside EVERY attention call: a cast of a
    [8192x4096] tensor = 134 MB written, per layer, per token -> ~8 GB/token of traffic for a static
    weight (~10 ms of the 112 ms budget). Fix: cast once at load (post-load hook). astype to the
    same dtype is a no-op and reshape is a view, so the per-call path becomes free. Numerically
    identical (same fp32 values). Cost: 40 x 134 MB = 5.4 GB resident."""
    return "wo_a_f32: registered (applies after load)"

POST_LOAD = {}
def _post_wo_a_f32(model):
    n = 0
    for layer in model.layers:
        w = layer.attn.wo_a.weight
        if w.dtype != mx.float32:
            layer.attn.wo_a.weight = w.astype(mx.float32); n += 1
    mx.eval([l.attn.wo_a.weight for l in model.layers])
    print(f"[patch] wo_a_f32: cast {n} wo_a weights to fp32 once ({n*0.134:.1f} GB resident)", flush=True)
POST_LOAD["wo_a_f32"] = _post_wo_a_f32


def _post_gate_f32(model):
    """Gate does `self.weight.astype(float32)` per call (384x5120 bf16 -> f32, 40x/token). Cast once."""
    n = 0
    for layer in model.layers:
        g = layer.ffn.gate
        if g.weight.dtype != mx.float32: g.weight = g.weight.astype(mx.float32); n += 1
    mx.eval([l.ffn.gate.weight for l in model.layers])
    print(f"[patch] gate_f32: {n} MoE gate weights cast to fp32 once", flush=True)
POST_LOAD["gate_f32"] = _post_gate_f32

def _post_shared_fuse(model):
    """SharedExpert runs w1 (gate) and w3 (up) as two quantized GEMVs on the same input. Fuse them
    into one QuantizedLinear of 2*inter rows (concatenate weight/scales/biases along the output
    axis) and split the result: one launch instead of two per layer, identical numerics."""
    from deepseek_v41_mlx import moe as MOE
    import mlx.nn as nn
    n = 0
    for layer in model.layers:
        se = layer.ffn.shared_experts
        w1, w3 = se.w1, se.w3
        if not isinstance(w1, nn.QuantizedLinear): continue
        fused = nn.QuantizedLinear(w1.weight.shape[1] * 32 // w1.bits, w1.weight.shape[0] * 2, bias=False,
                                   group_size=w1.group_size, bits=w1.bits)
        fused.weight = mx.concatenate([w1.weight, w3.weight], axis=0)
        fused.scales = mx.concatenate([w1.scales, w3.scales], axis=0)
        fused.biases = mx.concatenate([w1.biases, w3.biases], axis=0)
        se.w13 = fused; se.inter = w1.weight.shape[0]; n += 1
    mx.eval([l.ffn.shared_experts.w13.parameters() for l in model.layers if hasattr(l.ffn.shared_experts, "w13")])
    def call(self, x):
        if not hasattr(self, "w13"):
            return _orig_se_call(self, x)
        dtype = x.dtype
        gu = self.w13(x).astype(mx.float32)
        gate, up = gu[..., :self.inter], gu[..., self.inter:]
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit); gate = mx.minimum(gate, self.limit)
        return self.w2((nn.silu(gate) * up).astype(dtype))
    global _orig_se_call
    _orig_se_call = MOE.SharedExpert.__call__
    MOE.SharedExpert.__call__ = call
    print(f"[patch] shared_fuse: {n} shared-expert w1/w3 pairs fused into one GEMV", flush=True)
POST_LOAD["shared_fuse"] = _post_shared_fuse


def _post_freqs_pre(model):
    """Every Attention keeps its own RoPE cos/sin table and regrows it whenever end_pos exceeds
    its length -> during decode each of the 40 layers recomputes precompute_freqs_cis EVERY token
    (a small multi-launch graph, 40x/token). Precompute once for V41_FREQS positions (default
    16384; 2 x 4 MB per layer)."""
    import os
    n = int(os.environ.get("V41_FREQS", "16384"))
    for layer in model.layers: layer.attn._freqs(n)
    mx.eval([layer.attn._cos for layer in model.layers] + [layer.attn._sin for layer in model.layers])
    print(f"[patch] freqs_pre: RoPE tables precomputed to {n} positions for {len(model.layers)} layers", flush=True)
POST_LOAD["freqs_pre"] = _post_freqs_pre

def apply_post_load(model, names):
    """Call after load(model) for patches that need the weights (e.g. wo_a_f32)."""
    for n in names:
        n = n.strip()
        if n in POST_LOAD: POST_LOAD[n](model)

def patch_specdec():
    """Speculative decoding with a PROMPT-LOOKUP drafter (no draft model). Each step forwards
    [pending decided-but-uncached tokens] + [k drafted tokens] in ONE chunk-general forward,
    verifies the draft greedily against the returned logits and emits accepted + 1 bonus token.
    On rejection the cache is rolled back to the pre-step snapshot (port's _cache_snapshot /
    _cache_restore) and the accepted tokens are carried as `pending` into the NEXT forward, so a
    rejection never costs a separate re-forward. Per step: one n-token forward whose cost is ~one
    decode step while launch-bound. Drafter: match the trailing n-gram earlier in prompt+output."""
    from deepseek_v41_mlx import generate as G
    from deepseek_v41_mlx.generate import _forward, _cache_snapshot, _cache_restore
    import os
    K = int(os.environ.get("V41_SPEC_K", "8")); NG = int(os.environ.get("V41_SPEC_NGRAM", "3"))
    def lookup(seq, k):
        for n in range(NG, 1, -1):
            if len(seq) <= n: continue
            tail = seq[-n:]
            for s in range(len(seq) - n - 1, -1, -1):
                if seq[s:s + n] == tail:
                    cand = seq[s + n:s + n + k]
                    if cand: return cand
        return []
    STATS = {"steps": 0, "drafted": 0, "accepted": 0, "nodraft": 0, "rejects": 0}
    G.SPEC_STATS = STATS
    def greedy_generate(model, input_ids, max_new_tokens=64, max_seq_len=None, eos_id=1,
                        dtype=mx.float32, prefill_chunk=512):
        try: mx.set_wired_limit(int(470e9))
        except Exception: pass
        for k in STATS: STATS[k] = 0
        seq = list(input_ids) if not hasattr(input_ids[0], "__len__") else list(input_ids[0])
        ids = mx.array([seq])
        cache = model.make_cache(bsz=1, max_seq_len=max_seq_len or (len(seq) + max_new_tokens + K + 8), dtype=dtype)
        logits = None
        for a in range(0, ids.shape[1], prefill_chunk):
            logits = _forward(model, ids[:, a:a + prefill_chunk], cache)
        first = int(mx.argmax(logits[:, -1], axis=-1)[0])
        out = [first]            # every decided token (pending is always a suffix of out)
        pending = [first]        # decided but not yet in the cache
        while len(out) < max_new_tokens and out[-1] != eos_id:
            draft = lookup(seq + out, min(K, max_new_tokens - len(out)))
            inp = pending + draft
            STATS["steps"] += 1; STATS["drafted"] += len(draft)
            if not draft: STATS["nodraft"] += 1
            snap = _cache_snapshot(cache) if draft else None
            lg = model(mx.array([inp]), cache, last_logit_only=False)
            preds = mx.argmax(lg[0], axis=-1).tolist()          # preds[i] = token following inp[i]
            P = len(pending); acc = 0
            for j, d in enumerate(draft):
                if preds[P - 1 + j] == d: acc += 1
                else: break
            STATS["accepted"] += acc
            new = draft[:acc] + [preds[P - 1 + acc]]
            out += new
            if acc == len(draft):
                pending = [new[-1]]
            else:                                                # cache holds wrong tokens: roll back
                STATS["rejects"] += 1
                _cache_restore(cache, snap)
                pending = pending + new
        if eos_id in out: out = out[:out.index(eos_id)]
        return out[:max_new_tokens]
    G.greedy_generate = greedy_generate
    return f"specdec: prompt-lookup speculative decoding (K={K}, ngram={NG})"

PATCHES = {"sinkhorn": patch_sinkhorn, "sinkhorn_metal": patch_sinkhorn_metal, "async": patch_async,
           "wo_a_f32": patch_wo_a_f32, "specdec": patch_specdec,
           "hc_metal": patch_hc_metal, "rmsnorm": patch_rmsnorm, "idxcache": patch_idxcache,
           "fq_metal": patch_fq_metal, "rope_metal": patch_rope_metal, "nosync": patch_nosync,
           "sa_sdpa": patch_sa_sdpa, "sa_sdpa_bf16": patch_sa_sdpa_bf16,
           "sa_metal": patch_sa_metal, "hcmix_metal": patch_hcmix_metal, "gate_metal": patch_gate_metal}
DEFAULT_STACK = "sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa,nosync,async"
DEFAULT_POST = "wo_a_f32,gate_f32"   # post-load patches (apply_post_load(model, DEFAULT_POST.split(",")))

def apply(names):
    for n in names:
        n = n.strip()
        if not n: continue
        print(f"[patch] {PATCHES[n]()}", flush=True)
