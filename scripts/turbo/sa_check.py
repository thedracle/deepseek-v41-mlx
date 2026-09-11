"""Numerics + speed of the generalized sa_sdpa vs the port's sparse_attn at m=1 and m=6 (synthetic)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import sys, time; 
import mlx.core as mx
from deepseek_v41_mlx import sparse_attention as SA
orig = SA.sparse_attn
from deepseek_v41_mlx.turbo import patches as VP; VP.PATCHES["sa_sdpa"]()
new = SA.sparse_attn
for m, k in ((1, 128), (6, 128), (6, 640), (13, 640)):
    b, h, d, n = 1, 64, 512, 2048
    q = mx.random.normal((b, m, h, d)).astype(mx.bfloat16); kv = mx.random.normal((b, n, d)).astype(mx.bfloat16)
    sink = mx.random.normal((h,)); idx = mx.random.randint(0, n, (b, m, k)).astype(mx.int32)
    idx = mx.where(mx.random.uniform(shape=idx.shape) < 0.1, -1, idx)
    a = orig(q, kv, sink, idx, d ** -0.5); c = new(q, kv, sink, idx, d ** -0.5); mx.eval(a, c)
    diff = float(mx.max(mx.abs(a.astype(mx.float32) - c.astype(mx.float32))))
    def tm(f):
        for _ in range(3): mx.eval(f(q, kv, sink, idx, d ** -0.5))
        t = time.perf_counter()
        for _ in range(20): mx.eval(f(q, kv, sink, idx, d ** -0.5))
        return (time.perf_counter() - t) / 20 * 1000
    print(f"[sa] m={m:2d} k={k:4d}: max|diff| {diff:.2e}  port {tm(orig):.3f} ms  sdpa {tm(new):.3f} ms", flush=True)
