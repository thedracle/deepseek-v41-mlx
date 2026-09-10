"""Per-expert REAP saliency for DeepSeek-V4.1-Flash, from the resident
mixed-4_8bit build (427 GB — loads and runs; ppl proved it).

Saliency = mean over routed tokens of applied_routing_weight x ||expert_output||_2,
accumulated in two disjoint halves of the calibration set. The weight used is
the one actually applied to the expert output — post sqrt-softplus scoring,
post top-k normalization (sum + 1e-20) and the x1.5 routed scaling factor;
`gate.bias`/`bias_vl` shift selection only and never touch the weight, so they
play no part in saliency. All 40 layers are MoE (V4's hash layers are gone).

The MoE forward is instrumented by monkeypatching deepseek_v41_mlx.moe.MoE
(no mx.compile anywhere in this runtime, so nothing to disable). Sequences run
through the normal cache path in 512-token chunks — no giant graphs (Metal
watchdog), engram id history threaded exactly as in generation/ppl.

    .venv/bin/python scripts/reap_calibrate.py <BUILD> <calib_corpus.npy> <saliency.npz> [samples] [seq_len]
"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from deepseek_v41_mlx import moe as M
from deepseek_v41_mlx.load import load

BUILD = sys.argv[1]
IDS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "calib_corpus.npy")
OUT = sys.argv[3] if len(sys.argv) > 3 else "saliency.npz"
SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else 32
SEQ = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
CHUNK = int(os.environ.get("DSV41_PPL_CHUNK", "512"))


def main():
    model, args = load(BUILD)
    E, nL = args.n_routed_experts, args.n_layers
    for i, layer in enumerate(model.layers):
        layer.ffn._reap_idx = i

    sal = np.zeros((2, nL, E))
    cnt = np.zeros((2, nL, E))
    half = [0]

    def instrumented(self, x):
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf)                 # applied post-norm weights
        y = self.experts(xf, indices)                    # [tokens, topk, dim]
        yf = y.astype(mx.float32)
        contrib = weights * mx.sqrt(mx.sum(yf * yf, axis=-1))
        fi = indices.reshape(-1)
        s = mx.zeros((E,), dtype=mx.float32).at[fi].add(contrib.reshape(-1))
        c = mx.zeros((E,), dtype=mx.float32).at[fi].add(
            mx.ones(fi.shape, dtype=mx.float32))
        mx.eval(s, c)
        sal[half[0], self._reap_idx] += np.array(s, dtype=np.float64)
        cnt[half[0], self._reap_idx] += np.array(c, dtype=np.float64)
        out = mx.sum(yf * weights[..., None], axis=-2)
        out = out + self.shared_experts(xf).astype(mx.float32)
        return out.reshape(shape).astype(x.dtype)

    M.MoE.__call__ = instrumented

    ids = np.load(IDS)[: SAMPLES * SEQ].reshape(SAMPLES, SEQ)
    print(f"calibration: {SAMPLES} x {SEQ} = {ids.size} tokens, {nL} MoE layers x {E} experts, "
          f"chunk {CHUNK}", flush=True)
    t0 = time.time()
    for s_i in range(SAMPLES):
        half[0] = 0 if s_i < SAMPLES // 2 else 1
        cache = model.make_cache(bsz=1, max_seq_len=SEQ + 8, dtype=mx.bfloat16)
        row = mx.array(ids[s_i:s_i + 1])
        for a in range(0, SEQ, CHUNK):
            lg = model(row[:, a:a + CHUNK], cache, last_logit_only=True)
            mx.eval(lg)
        del cache
        mx.clear_cache()
        if (s_i + 1) % 4 == 0 or s_i == SAMPLES - 1:
            el = time.time() - t0
            print(f"  {s_i + 1}/{SAMPLES} sequences  ({el:.0f}s, "
                  f"{(s_i + 1) * SEQ / el:.0f} tok/s, peak {mx.get_peak_memory() / 1e9:.0f} GB)",
                  flush=True)

    total, count = sal.sum(0), cnt.sum(0)
    mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    halves = np.where(cnt > 0, sal / np.maximum(cnt, 1), 0.0)
    np.savez(OUT, saliency=mean, total=total, counts=count,
             saliency_halves=halves, counts_halves=cnt,
             tokens=np.array(ids.size), samples=np.array(SAMPLES),
             seq_len=np.array(SEQ), moe_layers=np.arange(nL))
    never = int((count == 0).sum())
    print(f"wrote {OUT} ({(time.time() - t0) / 60:.1f} min); "
          f"never-routed experts: {never}/{nL * E}", flush=True)

    print(f"\nsplit-half agreement:\n{'keep':>6} {'overlap':>9} {'spearman':>9}")
    for keep in (0.75, 0.63, 0.5):
        k = max(1, int(round(E * keep)))
        ov, rh = [], []
        for i in range(nL):
            a, b = halves[0, i], halves[1, i]
            ov.append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            rh.append(np.corrcoef(ra, rb)[0, 1])
        print(f"{keep:>6.0%} {np.mean(ov):>8.1%} {np.mean(rh):>9.3f}", flush=True)


if __name__ == "__main__":
    main()
