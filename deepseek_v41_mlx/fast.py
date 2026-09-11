"""Fast paths for the DeepSeek-V4.1 MLX runtime: native Metal kernels and mx.fast ops that
replace multi-launch elementwise/reduction chains. Decode is launch-bound (~6,500 kernel
launches per token on an M3 Ultra), so fusing these chains is what moves tok/s.

Every kernel is a drop-in for the Python function it replaces, verified against it on random
tensors (fake-quant bit-exact; the rest at fp32 ulp level) and by tests/test_fast.py running the
full reference parity battery with the fast paths on. ``DSV41_FAST=0`` (or ``fast.enable(False)``)
restores the pure-MLX reference path at every call site — the guards branch at call time, so
nothing is rebound.

Kept opt-in because it loses in situ: ``sparse_attn4`` (tiled prefill attention; wins a
random-index micro-bench, loses to the port's batched-GEMM einsum on real index patterns).
"""
import os
import mlx.core as mx

ENABLED = os.environ.get("DSV41_FAST", "1") != "0"
_HC = 4                      # the Sinkhorn kernel is specialised for hc_mult=4 (V4.1's value)
GATE_KERNEL = os.environ.get("DSV41_GATE_KERNEL", "0") == "1"   # K1-lite fused gate chain: opt-in, no in-situ gain (9.9 vs 9.8 tok/s)


def enable(flag: bool = True):
    global ENABLED
    ENABLED = bool(flag)


_SRC = r"""
    uint tid = thread_position_in_grid.x;
    if (tid >= n_tok) return;
    const uint HC = 4;
    const uint M = 2*HC + HC*HC;   // 24
    const device float* m = mixes + tid * M;
    float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

    // pre / post
    for (uint i = 0; i < HC; ++i) {
        float a = m[i] * s0 + hc_base[i];
        pre[tid*HC + i] = 1.0f / (1.0f + metal::exp(-a)) + eps;
        float b = m[HC + i] * s1 + hc_base[HC + i];
        post[tid*HC + i] = 2.0f / (1.0f + metal::exp(-b));
    }

    // comb: softmax over rows (last axis), +eps, then Sinkhorn sweeps
    float c[16];
    for (uint r = 0; r < HC; ++r) {
        float mx_ = -INFINITY;
        for (uint k = 0; k < HC; ++k) { c[r*HC+k] = m[2*HC + r*HC + k] * s2 + hc_base[2*HC + r*HC + k]; mx_ = metal::max(mx_, c[r*HC+k]); }
        float sum = 0.0f;
        for (uint k = 0; k < HC; ++k) { c[r*HC+k] = metal::exp(c[r*HC+k] - mx_); sum += c[r*HC+k]; }
        for (uint k = 0; k < HC; ++k) { c[r*HC+k] = c[r*HC+k] / sum + eps; }
    }
    // first: column-normalise (axis=-2), then (iters-1) x [row-normalise, column-normalise]
    for (uint k = 0; k < HC; ++k) { float cs = eps; for (uint r = 0; r < HC; ++r) cs += c[r*HC+k];
                                    for (uint r = 0; r < HC; ++r) c[r*HC+k] /= cs; }
    for (uint it = 0; it + 1 < iters; ++it) {
        for (uint r = 0; r < HC; ++r) { float rs = eps; for (uint k = 0; k < HC; ++k) rs += c[r*HC+k];
                                        for (uint k = 0; k < HC; ++k) c[r*HC+k] /= rs; }
        for (uint k = 0; k < HC; ++k) { float cs = eps; for (uint r = 0; r < HC; ++r) cs += c[r*HC+k];
                                        for (uint r = 0; r < HC; ++r) c[r*HC+k] /= cs; }
    }
    for (uint i = 0; i < 16; ++i) comb[tid*16 + i] = c[i];
"""

_kernel = mx.fast.metal_kernel(
    name="v41_sinkhorn_split",
    input_names=["mixes", "hc_scale", "hc_base", "n_tok", "iters", "eps"],
    output_names=["pre", "post", "comb"],
    source=_SRC,
)

def sinkhorn_split(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    assert hc_mult == _HC, "kernel specialised for hc_mult=4"
    lead = mixes.shape[:-1]
    n = 1
    for d in lead: n *= d
    m = mixes.astype(mx.float32).reshape(n, 2*_HC + _HC*_HC)
    pre, post, comb = _kernel(
        inputs=[m, hc_scale.astype(mx.float32), hc_base.astype(mx.float32),
                mx.array(n, dtype=mx.uint32), mx.array(sinkhorn_iters, dtype=mx.uint32),
                mx.array(eps, dtype=mx.float32)],
        grid=(n, 1, 1), threadgroup=(min(n, 64), 1, 1),
        output_shapes=[(n, _HC), (n, _HC), (n, 16)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return (pre.reshape(*lead, _HC), post.reshape(*lead, _HC), comb.reshape(*lead, _HC, _HC))


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "/tmp/dsv41")
    from deepseek_v41_mlx.hyper_connections import split_sinkhorn
    mx.random.seed(0)
    for shape in [(1, 1, 24), (1, 7, 24), (2, 5, 24)]:
        mixes = mx.random.normal(shape) * 2
        scale = mx.array([0.7, 1.3, 0.9]); base = mx.random.normal((24,)) * 0.5
        ref = split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
        got = sinkhorn_split(mixes, scale, base, 4, 20, 1e-6)
        mx.eval(*ref, *got)
        diffs = [float(mx.max(mx.abs(a - b))) for a, b in zip(ref, got)]
        print(f"  shape {shape}: max|diff| pre={diffs[0]:.2e} post={diffs[1]:.2e} comb={diffs[2]:.2e}")
    # timing at decode shape
    mixes = mx.random.normal((1, 1, 24)); scale = mx.ones((3,)); base = mx.zeros((24,))
    for fn, lbl in [(lambda: split_sinkhorn(mixes, scale, base, 4, 20, 1e-6), "eager"),
                    (mx.compile(lambda: split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)), "compiled"),
                    (lambda: sinkhorn_split(mixes, scale, base, 4, 20, 1e-6), "METAL kernel")]:
        for _ in range(10): mx.eval(*fn())
        t = time.perf_counter()
        for _ in range(200): mx.eval(*fn())
        dt = (time.perf_counter() - t) / 200 * 1000
        print(f"  {lbl:<14} {dt:7.3f} ms/call  -> {dt*80:6.1f} ms/token (80 calls)")


# ---------------------------------------------------------------------------
# hc_post / hc_pre: the hyper-connection collapse/expand, 80 calls each per token.
# hc_post: out[k,d] = post[k]*x[d] + sum_j comb[j,k]*res[j,d]   (residual on the SUMMED axis j)
# One thread per (k,d) output element, one launch, no [b,s,4,4,d] intermediate.
_HCPOST_SRC = r"""
    uint gid = thread_position_in_grid.x;            // over n*HC*D
    const uint HC = 4;
    uint d   = gid % D;
    uint k   = (gid / D) % HC;
    uint tok = gid / (D * HC);
    if (tok >= n_tok) return;
    float acc = post[tok*HC + k] * (float)x[tok*D + d];
    for (uint j = 0; j < HC; ++j)
        acc += comb[tok*HC*HC + j*HC + k] * (float)residual[(tok*HC + j)*D + d];
    out[gid] = (T)acc;
"""
_hcpost_kernel = mx.fast.metal_kernel(
    name="v41_hc_post", input_names=["x", "residual", "post", "comb", "n_tok", "D"],
    output_names=["out"], source=_HCPOST_SRC)

def hc_post(x, residual, post, comb):
    """x [b,s,d]; residual [b,s,hc,d]; post [b,s,hc]; comb [b,s,hc,hc] -> [b,s,hc,d] in x.dtype"""
    b, s, d = x.shape; n = b * s
    out, = _hcpost_kernel(
        inputs=[x.reshape(n, d), residual.reshape(n, _HC, d),
                post.astype(mx.float32).reshape(n, _HC), comb.astype(mx.float32).reshape(n, _HC*_HC),
                mx.array(n, dtype=mx.uint32), mx.array(d, dtype=mx.uint32)],
        template=[("T", x.dtype)],
        grid=(n*_HC*d, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n, _HC, d)], output_dtypes=[x.dtype])
    return out.reshape(b, s, _HC, d)

# hc_pre: y[d] = sum_k pre[k]*x[k,d]   (collapse the hc copies)
_HCPRE_SRC = r"""
    uint gid = thread_position_in_grid.x;            // over n*D
    const uint HC = 4;
    uint d = gid % D; uint tok = gid / D;
    if (tok >= n_tok) return;
    float acc = 0.0f;
    for (uint k = 0; k < HC; ++k) acc += pre[tok*HC + k] * (float)x[(tok*HC + k)*D + d];
    out[gid] = (T)acc;
"""
_hcpre_kernel = mx.fast.metal_kernel(
    name="v41_hc_pre", input_names=["x", "pre", "n_tok", "D"], output_names=["out"], source=_HCPRE_SRC)

def hc_pre(x, pre_mix):
    """x [b,s,hc,d]; pre_mix [b,s,hc] -> [b,s,d] in x.dtype"""
    b, s, hc, d = x.shape; n = b * s
    out, = _hcpre_kernel(
        inputs=[x.reshape(n, hc, d), pre_mix.astype(mx.float32).reshape(n, hc),
                mx.array(n, dtype=mx.uint32), mx.array(d, dtype=mx.uint32)],
        template=[("T", x.dtype)],
        grid=(n*d, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(n, d)], output_dtypes=[x.dtype])
    return out.reshape(b, s, d)


# ---------------------------------------------------------------------------
# rope_tail: adjacent-pair rotation on the LAST rd channels, copy the rest. ~136 calls/token,
# the port's version is ~12 launches (concat + astype + reshape + slices + 4 mul + 2 add + stack).
# One thread per (row, pair). x [rows, D]; rows = b*n*h (4D) or b*n (3D); pos(row) = (row // h) % n.
_ROPE_SRC = r"""
    uint gid = thread_position_in_grid.x;              // over rows * D/2
    uint hd2 = D / 2;
    uint p   = gid % hd2;
    uint row = gid / hd2;
    if (row >= rows) return;
    uint base = row * D + 2*p;
    float xr = (float)x[base], xi = (float)x[base + 1];
    uint tail0 = (D - rd) / 2;                         // first rotated pair
    if (p < tail0) { out[base] = (T)xr; out[base+1] = (T)xi; return; }
    uint pos = (row / h) % n;
    uint fi  = pos * (rd/2) + (p - tail0);
    float c = cosv[fi], s = sinv[fi] * sgn;
    out[base]   = (T)(xr * c - xi * s);
    out[base+1] = (T)(xr * s + xi * c);
"""
_rope_kernel = mx.fast.metal_kernel(
    name="v41_rope_tail", input_names=["x", "cosv", "sinv", "rows", "D", "rd", "h", "n", "sgn"],
    output_names=["out"], source=_ROPE_SRC)

def rope_tail(x, rd, cos, sin, inverse=False):
    """x [b,n,h,d] or [b,n,d]; cos/sin [n, rd//2]. Matches layers.rope_tail exactly (fp32 math, cast out)."""
    shp = x.shape; D = shp[-1]
    if x.ndim == 4: b, n, h = shp[0], shp[1], shp[2]
    else:           b, n, h = shp[0], shp[1], 1
    rows = b * n * h
    out, = _rope_kernel(
        inputs=[x.reshape(rows, D), cos.astype(mx.float32).reshape(-1), sin.astype(mx.float32).reshape(-1),
                mx.array(rows, mx.uint32), mx.array(D, mx.uint32), mx.array(rd, mx.uint32),
                mx.array(h, mx.uint32), mx.array(n, mx.uint32), mx.array(-1.0 if inverse else 1.0, mx.float32)],
        template=[("T", x.dtype)], grid=(rows * (D // 2), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, D)], output_dtypes=[x.dtype])
    return out.reshape(shp)


# ---------------------------------------------------------------------------
# fake_quant_fp8_ue8m0: the port's version is ~25 launches; 40 calls/token (window KV).
# One thread per 32-element block: amax -> ue8m0 pow2 scale (IEEE bit trick, exactly as
# fast_round_scale) -> per element clip(x/s) -> e4m3fn round-to-nearest-even -> *s.
# e4m3fn RNE is implemented explicitly (Metal has no fp8) and verified vs mx.to_fp8/from_fp8.
_FQ8_SRC = r"""
    uint blk = thread_position_in_grid.x;               // over n_blocks
    if (blk >= n_blocks) return;
    const uint B = 32;
    const device T* xp = x + blk * B;
    float amax = 0.0f;
    for (uint i = 0; i < B; ++i) amax = metal::max(amax, metal::abs((float)xp[i]));
    amax = metal::max(amax, 1e-4f);
    // ue8m0 scale: 2^ceil(log2(amax * (1/448)))
    float v = amax * (1.0f / 448.0f);
    uint bits = as_type<uint>(v);
    int e = (int)((bits >> 23) & 0xFFu) - 127;
    uint man = bits & 0x7FFFFFu;
    e += (man != 0u) ? 1 : 0;
    float s = as_type<float>((uint)(e + 127) << 23);
    for (uint i = 0; i < B; ++i) {
        float q = metal::clamp((float)xp[i] / s, -448.0f, 448.0f);
        // ---- e4m3fn round-to-nearest-even ----
        float a = metal::abs(q);
        float r;
        if (a == 0.0f) r = 0.0f;
        else if (a < 0.015625f) {                        // subnormal: step 2^-9
            r = metal::rint(a * 512.0f) * (1.0f / 512.0f);
        } else {
            int ex; float m = metal::frexp(a, ex);        // a = m * 2^ex, m in [0.5,1)
            ex -= 1; m *= 2.0f;                           // m in [1,2)
            float q8 = metal::rint((m - 1.0f) * 8.0f);    // 3 mantissa bits, RNE
            if (q8 >= 8.0f) { q8 = 0.0f; ex += 1; }
            r = (1.0f + q8 / 8.0f) * metal::exp2((float)ex);
            r = metal::min(r, 448.0f);
        }
        r = (q < 0.0f) ? -r : r;
        out[blk * B + i] = (T)(r * s);
    }
"""
_fq8_kernel = mx.fast.metal_kernel(
    name="v41_fq_fp8_ue8m0", input_names=["x", "n_blocks"], output_names=["out"], source=_FQ8_SRC)

def fake_quant_fp8_ue8m0(x, block=32):
    assert block == 32
    shp = x.shape; n = 1
    for d in shp: n *= d
    nb = n // 32
    out, = _fq8_kernel(inputs=[x.reshape(-1), mx.array(nb, mx.uint32)], template=[("T", x.dtype)],
                       grid=(nb, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[x.dtype])
    return out.reshape(shp)


# ---------------------------------------------------------------------------
# FP4 fake-quant (two variants). Shared helpers in a header: e4m3 RNE (verified bit-exact in the
# fp8 kernel above) and e2m1 RNE with the port's exact tie rule
# (ties at .25/1.25/2.5/5 round DOWN via '>', ties at .75/1.75/3.5 round UP via '>=').
_FQ4_HDR = r"""
inline float v41_e4m3_rne(float a) {                       // a >= 0
    if (a == 0.0f) return 0.0f;
    if (a < 0.015625f) return metal::rint(a * 512.0f) * (1.0f / 512.0f);
    int ex; float m = metal::frexp(a, ex); ex -= 1; m *= 2.0f;
    float q8 = metal::rint((m - 1.0f) * 8.0f);
    if (q8 >= 8.0f) { q8 = 0.0f; ex += 1; }
    return metal::min((1.0f + q8 / 8.0f) * metal::exp2((float)ex), 448.0f);
}
inline float v41_e2m1_rne(float v) {                       // |v| <= 6, keeps sign, sign(0)=0
    float a = metal::abs(v);
    int idx = (a > 0.25f) + (a > 1.25f) + (a > 2.5f) + (a > 5.0f) + (a >= 0.75f) + (a >= 1.75f) + (a >= 3.5f);
    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float r = lut[idx];
    return (v > 0.0f) ? r : ((v < 0.0f) ? -r : 0.0f);
}
inline float v41_ue8m0(float v) {                          // 2^ceil(log2 v), v>0, IEEE bit trick
    uint bits = as_type<uint>(v);
    int e = (int)((bits >> 23) & 0xFFu) - 127;
    e += ((bits & 0x7FFFFFu) != 0u) ? 1 : 0;
    return as_type<float>((uint)(e + 127) << 23);
}
"""
# fp4_ue8m0: block 32, amax floor 6*2^-126, scale = ue8m0(amax*(1/6))   [indexer q/k]
_FQ4U_SRC = r"""
    uint blk = thread_position_in_grid.x; if (blk >= n_blocks) return;
    const uint B = 32; const device T* xp = x + blk * B;
    float amax = 0.0f;
    for (uint i = 0; i < B; ++i) amax = metal::max(amax, metal::abs((float)xp[i]));
    amax = metal::max(amax, 6.0f * 1.1754943508222875e-38f);
    float s = v41_ue8m0(amax * (1.0f / 6.0f));
    for (uint i = 0; i < B; ++i) {
        float q = metal::clamp((float)xp[i] / s, -6.0f, 6.0f);
        out[blk * B + i] = (T)(v41_e2m1_rne(q) * s);
    }
"""
# fp4_e4m3: block 16, amax floor 6*2^-9, scale = e4m3_rne(amax / 6)   [compressed KV latents]
_FQ4E_SRC = r"""
    uint blk = thread_position_in_grid.x; if (blk >= n_blocks) return;
    const uint B = 16; const device T* xp = x + blk * B;
    float amax = 0.0f;
    for (uint i = 0; i < B; ++i) amax = metal::max(amax, metal::abs((float)xp[i]));
    amax = metal::max(amax, 6.0f * 0.001953125f);
    float s = v41_e4m3_rne(amax / 6.0f);
    for (uint i = 0; i < B; ++i) {
        float q = metal::clamp((float)xp[i] / s, -6.0f, 6.0f);
        out[blk * B + i] = (T)(v41_e2m1_rne(q) * s);
    }
"""
_fq4u_kernel = mx.fast.metal_kernel(name="v41_fq_fp4_ue8m0", input_names=["x","n_blocks"], output_names=["out"], header=_FQ4_HDR, source=_FQ4U_SRC)
_fq4e_kernel = mx.fast.metal_kernel(name="v41_fq_fp4_e4m3",  input_names=["x","n_blocks"], output_names=["out"], header=_FQ4_HDR, source=_FQ4E_SRC)

def _fq_run(kernel, x, block):
    shp = x.shape; n = 1
    for d in shp: n *= d
    nb = n // block
    out, = kernel(inputs=[x.reshape(-1), mx.array(nb, mx.uint32)], template=[("T", x.dtype)],
                  grid=(nb,1,1), threadgroup=(64,1,1), output_shapes=[(n,)], output_dtypes=[x.dtype])
    return out.reshape(shp)
def fake_quant_fp4_ue8m0(x, block=32): assert block == 32; return _fq_run(_fq4u_kernel, x, 32)
def fake_quant_fp4_e4m3(x, block=16):  assert block == 16; return _fq_run(_fq4e_kernel, x, 16)


# ---------------------------------------------------------------------------
# sparse_attn v4 (PREFILL): one 1024-thread threadgroup = 32 heads (one per simdgroup) of ONE query.
# Key rows are staged in threadgroup memory in tiles of 8 and read by all 32 heads -> each row is
# fetched from device memory once per 32 heads instead of once per head (v3). Online softmax per
# head in registers (lane owns the interleaved dims lane+32*i, conflict-free tile reads); the sink
# is the initial (max=sink, sum=1) state. fp32 throughout.
_SA4_SRC = r"""
    const uint TK = 8;
    threadgroup float tile[8 * 512];
    uint grp  = threadgroup_position_in_grid.y;              // r * (H/32) + half
    uint t    = thread_position_in_threadgroup.x;
    uint sg   = t / 32, lane = t % 32;
    uint hf = grp % (H / 32), r = grp / (H / 32), bi = r / M;
    uint hh   = hf * 32 + sg;
    const device Tq* qp = q + (r*H + hh) * D;
    float qr[16], acc[16];
    for (uint i = 0; i < 16; ++i) { qr[i] = (float)qp[i*32 + lane] * scale; acc[i] = 0.0f; }
    float mx_ = sink[hh], sm = 1.0f;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // cooperative tile load: 8 rows x 512 = 4096 floats / 1024 threads
        for (uint e = t; e < TK * D; e += 1024) {
            uint kk = k0 + e / D, dd = e % D;
            float v = 0.0f;
            if (kk < K) { int j = idx[r*K + kk]; if (j >= 0) v = (float)kv[((uint)bi * N + (uint)j) * D + dd]; }
            tile[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lg[8];
        float tmax = -1e30f;
        for (uint u = 0; u < TK; ++u) {
            uint kk = k0 + u;
            float p = 0.0f;
            for (uint i = 0; i < 16; ++i) p += qr[i] * tile[u*D + i*32 + lane];
            p = metal::simd_sum(p);
            bool valid = (kk < K) && (idx[r*K + kk] >= 0);
            lg[u] = valid ? p : -1e30f;
            tmax = metal::max(tmax, lg[u]);
        }
        float mnew = metal::max(mx_, tmax);
        float corr = metal::exp(mx_ - mnew);
        sm *= corr;
        for (uint i = 0; i < 16; ++i) acc[i] *= corr;
        for (uint u = 0; u < TK; ++u) {
            float w = (lg[u] <= -1e30f) ? 0.0f : metal::exp(lg[u] - mnew);
            sm += w;
            if (w != 0.0f) for (uint i = 0; i < 16; ++i) acc[i] += w * tile[u*D + i*32 + lane];
        }
        mx_ = mnew;
    }
    for (uint i = 0; i < 16; ++i) out[(r*H + hh) * D + i*32 + lane] = (To)(acc[i] / sm);
"""
_sa4_kernel = mx.fast.metal_kernel(
    name="v41_sparse_attn4", input_names=["q", "kv", "sink", "idx", "R", "M", "H", "D", "N", "K", "scale"],
    output_names=["out"], source=_SA4_SRC)

def sparse_attn4(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
    b, m, h, d = q.shape; n = kv.shape[1]; k = topk_idxs.shape[-1]
    assert d == 512 and h % 32 == 0, "v4 specialised for D=512, H multiple of 32"
    R = b * m
    out, = _sa4_kernel(
        inputs=[q.reshape(R*h, d), kv.reshape(b*n, d), attn_sink.astype(mx.float32),
                topk_idxs.astype(mx.int32).reshape(R, k),
                mx.array(R, mx.uint32), mx.array(m, mx.uint32), mx.array(h, mx.uint32),
                mx.array(d, mx.uint32), mx.array(n, mx.uint32), mx.array(k, mx.uint32), mx.array(softmax_scale, mx.float32)],
        template=[("Tq", q.dtype), ("Tk", kv.dtype), ("To", q.dtype)],
        grid=(1024, R * (h // 32), 1), threadgroup=(1024, 1, 1),
        output_shapes=[(R*h, d)], output_dtypes=[q.dtype])
    return out.reshape(b, m, h, d)


# ---------------------------------------------------------------------------
# Sparse attention via mx.fast.scaled_dot_product_attention. The per-head attention sink becomes
# ONE extra key with a zero value vector whose logit is attn_sink[h] via the additive mask
# (identical to the reference's denominator-only sink by softmax shift invariance); masked
# indices -> -1e30. MLA's single KV head is SDPA's GQA fast path. m == 1 (decode): 1 SDPA call.
# m > 1: one SDPA batch row per query up to m*k = 4096 gathered keys (verify chunks); beyond that
# the reference einsum chain is faster (measured), so the caller's fallback is used.
SA_PREFILL = os.environ.get("DSV41_SA_PREFILL", "ref")      # "v4" = sparse_attn4 for m*k > 4096


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk, fallback):
    b, m, h, d = q.shape
    n = kv.shape[1]; k = topk_idxs.shape[-1]
    if m > 1 and m * k > 4096:
        if SA_PREFILL == "v4" and d == 512 and h % 32 == 0:
            return sparse_attn4(q, kv, attn_sink, topk_idxs, softmax_scale)
        return fallback(q, kv, attn_sink, topk_idxs, softmax_scale, chunk)
    safe = mx.maximum(topk_idxs, 0).astype(mx.int32)
    base = (mx.arange(b, dtype=mx.int32) * n).reshape(b, 1, 1)
    g = kv.reshape(b * n, d)[(safe + base).reshape(-1)].reshape(b * m, k, d)
    kvx = mx.concatenate([g, mx.zeros((b * m, 1, d), dtype=g.dtype)], axis=1)[:, None].astype(mx.float32)
    Q = q.reshape(b * m, h, 1, d).astype(mx.float32)
    mask = mx.where(topk_idxs.reshape(b * m, 1, 1, k) >= 0, 0.0, -1e30).astype(mx.float32)
    mask = mx.concatenate([mx.broadcast_to(mask, (b * m, h, 1, k)),
                           mx.broadcast_to(attn_sink.astype(mx.float32).reshape(1, h, 1, 1), (b * m, h, 1, 1))], axis=-1)
    o = mx.fast.scaled_dot_product_attention(Q, kvx, kvx, scale=softmax_scale, mask=mask)
    return o.reshape(b, m, h, d).astype(q.dtype)


# ---------------------------------------------------------------------------
# MoE gate post-matmul chain (K1-lite). The 384x5120 matmul stays MLX's; only what follows is fused.
_GATE_SRC = r"""
    threadgroup float sc[1024];      // scores (unbiased, for the weights)
    threadgroup float bs[1024];      // biased (for selection)
    threadgroup float red_v[8]; threadgroup int red_i[8];
    threadgroup int sel[16];
    uint t = thread_position_in_threadgroup.x, tid = threadgroup_position_in_grid.y;
    uint sg = t / 32, lane = t % 32;
    for (uint e = t; e < E; e += 256) {
        float z = (float)logits[tid * E + e] / temp;
        float sp = z > 20.0f ? z : metal::log(1.0f + metal::exp(z));   // softplus, stable
        float s = metal::sqrt(sp);
        sc[e] = s; bs[e] = s + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wsum = 0.0f;
    for (uint r = 0; r < K; ++r) {
        float bv = -1e30f; int bi = -1;
        for (uint e = t; e < E; e += 256) { float v = bs[e]; if (v > bv || (v == bv && (int)e < bi)) { bv = v; bi = (int)e; } }
        // simd reduce (max value, lowest index on ties)
        for (uint o = 16; o > 0; o >>= 1) {
            float ov = metal::simd_shuffle_down(bv, o); int oi = metal::simd_shuffle_down(bi, o);
            if (ov > bv || (ov == bv && oi >= 0 && (oi < bi || bi < 0))) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[sg] = bv; red_i[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float mv = -1e30f; int mi = -1;
            for (uint g = 0; g < 8; ++g) { if (red_v[g] > mv || (red_v[g] == mv && red_i[g] >= 0 && (red_i[g] < mi || mi < 0))) { mv = red_v[g]; mi = red_i[g]; } }
            sel[r] = mi; bs[mi] = -1e30f;                      // remove from further rounds
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t < K) { int e = sel[t]; float w = sc[e]; wsum = w; }
    // normalise over the K selected (K <= 32: one simd group)
    float tot = metal::simd_sum(sg == 0 ? wsum : 0.0f);
    if (t < K) {
        int e = sel[t]; float w = sc[e];
        if (NORM != 0) w = w / (tot + 1e-20f);
        out_w[tid * K + t] = w * route_scale;
        out_i[tid * K + t] = e;
    }
"""
_gate_kernel = mx.fast.metal_kernel(name="v41_gate_topk", input_names=["logits", "bias", "temp", "route_scale", "E", "K", "NORM"],
                               output_names=["out_w", "out_i"], source=_GATE_SRC)
def gate_topk(logits, bias, k, temp, route_scale, norm):
    """MoE gate post-matmul chain fused: /temp -> sqrt(softplus) -> +bias -> top-k -> gather scores ->
    normalise -> route_scale, one launch per call instead of ~8 (sqrtsoftplus gates, E <= 1024, k <= 16).
    Same expert sets as argpartition, weights within 1e-7. 320 -> 241 us at M=1 (synthetic)."""
    T, E = logits.shape
    w, i = _gate_kernel(inputs=[logits, bias.astype(mx.float32), mx.array(temp, mx.float32), mx.array(route_scale, mx.float32),
                           mx.array(E, mx.uint32), mx.array(k, mx.uint32), mx.array(1 if norm else 0, mx.uint32)],
                   grid=(256, T, 1), threadgroup=(256, 1, 1), output_shapes=[(T, k), (T, k)], output_dtypes=[mx.float32, mx.int32])
    return w, i

