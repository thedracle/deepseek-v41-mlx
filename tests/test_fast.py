"""Fast-path parity: every kernel in deepseek_v41_mlx/fast.py against the pure-MLX function it
replaces (random tensors at the tiny-config AND the release shapes), then the full reference
battery from tests/test_parity.py with the fast paths ON. Fake-quant kernels must be bit-exact;
the rest are fp32-ulp level (the reference battery's own thresholds decide what is acceptable).

    .venv/bin/python tests/test_fast.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mlx.core as mx  # noqa: E402
from deepseek_v41_mlx import fast, fakequant as FQ, hyper_connections as HC, layers as L, sparse_attention as SA  # noqa: E402


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-9)


def both(fn, *args, **kw):
    """Run fn with the fast path off, then on."""
    fast.enable(False); ref = fn(*args, **kw); mx.eval(ref)
    fast.enable(True); out = fn(*args, **kw); mx.eval(out)
    return ref, out


def unit_parity():
    mx.random.seed(0)
    results = {}
    # Sinkhorn split (hc_mult=4): fp32, 20 sweeps
    for shape in [(1, 1, 24), (2, 7, 24), (1, 512, 24)]:
        mixes = mx.random.normal(shape) * 2
        scale = mx.array([0.7, 1.3, 0.9]); base = mx.random.normal((24,)) * 0.5
        r, o = both(HC.split_sinkhorn, mixes, scale, base, 4, 20, 1e-6)
        results[f"sinkhorn {shape}"] = max(rel(o[i], r[i]) for i in range(3))
    # hc_pre / hc_post
    for (b, s, d) in [(1, 1, 64), (2, 5, 64), (1, 6, 5120)]:
        x = mx.random.normal((b, s, 4, d)).astype(mx.bfloat16); pre = mx.softmax(mx.random.normal((b, s, 4)), axis=-1)
        r, o = both(HC.hc_pre, x, pre); results[f"hc_pre {(b, s, d)}"] = rel(o.astype(mx.float32), r.astype(mx.float32))
        y = mx.random.normal((b, s, d)).astype(mx.bfloat16); post = mx.random.normal((b, s, 4)); comb = mx.softmax(mx.random.normal((b, s, 4, 4)), axis=-1)
        r, o = both(HC.hc_post, y, x, post, comb); results[f"hc_post {(b, s, d)}"] = rel(o.astype(mx.float32), r.astype(mx.float32))
    # RMSNorm (fp32 reduction) and rope_tail
    for d in (64, 512):
        n = L.RMSNorm(d); n.weight = 1 + 0.1 * mx.random.normal((d,))
        x = mx.random.normal((2, 3, d)).astype(mx.bfloat16)
        r, o = both(n, x); results[f"rmsnorm d={d}"] = rel(o.astype(mx.float32), r.astype(mx.float32))
        rd = 16 if d == 64 else 64
        cos, sin = L.precompute_freqs_cis(rd, 40, 64, 10000.0, 4.0, 32, 1)
        for shape in [(1, 3, d), (2, 3, 4, d)]:
            x = mx.random.normal(shape).astype(mx.bfloat16)
            pos = cos[5:8], sin[5:8]
            for inv in (False, True):
                r, o = both(L.rope_tail, x, rd, pos[0], pos[1], inv)
                results[f"rope_tail {shape} inv={inv}"] = rel(o.astype(mx.float32), r.astype(mx.float32))
    # fake quant: BIT-EXACT, including midpoints and power-of-two scales
    rng = np.random.default_rng(3)
    xs = [(rng.standard_normal((256, 64)) * np.exp(rng.standard_normal((256, 1)))).astype(np.float32),
          (np.array([0.033203125, -0.033203125, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0, 448.0, 6.0, 1e-9, 0.0, -0.0, 2.0] * 8,
                    dtype=np.float32).reshape(2, 64) * 2.0 ** rng.integers(-8, 9, size=(2, 1)).astype(np.float32)),
          rng.standard_normal((4, 7, 512)).astype(np.float32)]
    for x in xs:
        for fn, blk in ((FQ.fake_quant_fp8_ue8m0, 32), (FQ.fake_quant_fp4_ue8m0, 32), (FQ.fake_quant_fp4_e4m3, 16)):
            r, o = both(fn, mx.array(x), blk)
            assert np.array_equal(np.array(r), np.array(o)), f"{fn.__name__} not bit-exact"
    results["fake-quant (3 formats, 3 tensors)"] = 0.0
    # sparse attention: SDPA-with-sink path (m=1 decode, m=6 verify) vs the reference chain
    for (m, k, n, d, h) in [(1, 12, 64, 64, 4), (6, 12, 64, 64, 4), (1, 640, 4096, 512, 64), (6, 640, 4096, 512, 64), (13, 640, 4096, 512, 64)]:
        q = mx.random.normal((1, m, h, d)).astype(mx.bfloat16); kv = mx.random.normal((1, n, d)).astype(mx.bfloat16)
        sink = mx.random.normal((h,)); idx = mx.random.randint(0, n, (1, m, k)).astype(mx.int32)
        idx = mx.where(mx.random.uniform(shape=idx.shape) < 0.1, -1, idx)
        r, o = both(SA.sparse_attn, q, kv, sink, idx, d ** -0.5)
        results[f"sparse_attn m={m} k={k} d={d}"] = rel(o.astype(mx.float32), r.astype(mx.float32))
    fast.enable(True)
    print("fast-path unit parity (rel max diff vs pure MLX):")
    for k, v in results.items():
        print(f"  {k:40s} {v:.3e}")
    worst = max(v for k, v in results.items() if not k.startswith("sparse_attn"))
    # fp32 kernels are exact or ulp-level; the bf16-I/O cases (hc_post at d=5120) differ by one bf16
    # rounding of the output where the fp32 accumulation order differs: 1e-4 relative, expected.
    assert worst < 5e-4, f"fast-path unit parity: {worst}"
    assert max(v for k, v in results.items() if k.startswith("sparse_attn")) < 5e-3, "sparse_attn parity (bf16 inputs)"
    print("  unit parity OK")


if __name__ == "__main__":
    unit_parity()
    print("\nreference battery with fast paths ON (tests/test_parity.py):")
    fast.enable(True)
    import test_parity  # noqa: E402
    test_parity.main()
