"""Torch stand-ins for the reference's TileLang kernels, so model.py runs on CPU.

Written from the kernel semantics in docs/reference/kernel.py — block absolute
max, power-of-two scale rounding via IEEE bit manipulation (fast_round_scale),
e4m3 round-trip through torch's native cast, e2m1 rounding to nearest-even on
the {0,.5,1,1.5,2,3,4,6} grid — not from the MLX port, so a parity test using
them compares two independent implementations.

Only the pure-math paths are provided. ``fp8_gemm`` / ``fp4_gemm`` raise: the
parity tests run the reference with bf16/fp32 weights, which never reach them.
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# Mirrors deepseek_v41_mlx.fakequant.DISABLE: parity runs compare continuous
# math with the QAT simulation off on both sides, then the full path separately.
DISABLE_FAKE_QUANT = False


def _pow2_ceil_log2(r: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(r)) via IEEE 754 bits, matching fast_round_scale exactly."""
    r = r.float().contiguous()
    bits = r.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    e = exp + (man != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def _e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Round |v| <= 6 to nearest e2m1, ties to even, keeping the sign."""
    mag = v.abs()
    idx = torch.zeros_like(mag, dtype=torch.long)
    for t in [0.25, 1.25, 2.5, 5.0]:       # tie rounds DOWN (even code below)
        idx = idx + (mag > t).long()
    for t in [0.75, 1.75, 3.5]:            # tie rounds UP (even code above)
        idx = idx + (mag >= t).long()
    return torch.sign(v) * _E2M1[idx]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    """Block-wise FP8 e4m3 quantization; inplace=True is fused quant+dequant."""
    if DISABLE_FAKE_QUANT and inplace:
        return x
    orig = x.shape
    xb = x.float().reshape(*orig[:-1], orig[-1] // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _pow2_ceil_log2(amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))
    else:
        s = amax / FP8_MAX
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    q = q.reshape(orig).to(x.dtype)
    if inplace:
        x.copy_(q)
        return x
    return q, s.squeeze(-1)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 e2m1; e8m0 (pow2) or e4m3 scales; inplace fused round-trip."""
    if DISABLE_FAKE_QUANT and inplace:
        return x
    orig = x.shape
    xb = x.float().reshape(*orig[:-1], orig[-1] // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6.0 * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6.0 * 2.0 ** -126)
        s = _pow2_ceil_log2(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    q = _e2m1_round((xb / s).clamp(-FP4_MAX, FP4_MAX)) * s
    q = q.reshape(orig).to(x.dtype)
    if inplace:
        x.copy_(q)
        return x
    return q, s.squeeze(-1)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gathered sparse attention with the sink in the denominator only.

    q [b,m,h,d], kv [b,n,d] (shared K=V), topk_idxs [b,m,k] int32, -1 masked.
    Max floored at -1e30 so an all-masked row yields zeros, as in the kernel.
    """
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    safe = idx.clamp(min=0)
    kvg = torch.gather(kv.float().unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                       safe.unsqueeze(-1).expand(b, m, idx.size(-1), d))
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), kvg) * softmax_scale
    valid = (idx >= 0).unsqueeze(2)
    logits = torch.where(valid, logits, torch.tensor(float("-inf")))
    mmax = logits.amax(-1, keepdim=True).clamp_min(-1e30)
    w = torch.exp(logits - mmax)
    w = torch.where(valid, w, torch.tensor(0.0))
    sink = attn_sink.float().view(1, 1, h, 1)
    denom = w.sum(-1, keepdim=True) + torch.exp(sink - mmax)
    o = torch.einsum("bmhk,bmkd->bmhd", w, kvg) / denom
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Transcription of hc_split_sinkhorn_kernel: sigmoid pre (+eps), 2*sigmoid
    post, comb row-softmax + eps then alternating col/row normalization."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _unavailable(name):
    def f(*a, **k):
        raise RuntimeError(f"{name} is a TileLang kernel and is not stubbed")
    return f


fp8_gemm = _unavailable("fp8_gemm")
fp4_gemm = _unavailable("fp4_gemm")
