"""Prefill-shape sparse attention: port gather path vs my Metal split-K kernel vs SDPA rows."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import sys, time; 
import mlx.core as mx
from deepseek_v41_mlx import sparse_attention as SA
import v41_metal as VM
orig = SA.sparse_attn
def tm(f, *a):
    for _ in range(2): mx.eval(f(*a))
    t = time.perf_counter()
    for _ in range(5): mx.eval(f(*a))
    return (time.perf_counter() - t) / 5 * 1000
for m, k, n in ((128, 128, 4096), (512, 128, 4096), (512, 640, 4096), (512, 640, 16384), (2048, 640, 16384)):
    b, h, d = 1, 64, 512
    q = mx.random.normal((b, m, h, d)).astype(mx.bfloat16); kv = mx.random.normal((b, n, d)).astype(mx.bfloat16)
    sink = mx.random.normal((h,)); idx = mx.random.randint(0, n, (b, m, k)).astype(mx.int32)
    idx = mx.where(mx.random.uniform(shape=idx.shape) < 0.1, -1, idx)
    a = orig(q, kv, sink, idx, d ** -0.5)
    res = {"port": tm(orig, q, kv, sink, idx, d ** -0.5)}
    for name in ("sparse_attn3", "sparse_attn2", "sparse_attn"):
        f = getattr(VM, name, None)
        if f is None: continue
        try:
            c = f(q, kv, sink, idx, d ** -0.5); mx.eval(c)
            diff = float(mx.max(mx.abs(a.astype(mx.float32) - c.astype(mx.float32))))
            res[name] = f"{tm(f, q, kv, sink, idx, d ** -0.5):.2f}ms(diff {diff:.1e})"
        except Exception as e: res[name] = f"ERR {str(e)[:60]}"
    print(f"[pf] m={m:4d} k={k:3d} n={n:5d}: port {res['port']:.2f} ms | " + " | ".join(f"{k2} {v}" for k2, v in res.items() if k2 != "port"), flush=True)
