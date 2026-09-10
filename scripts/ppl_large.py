"""Perplexity over the shared held-out corpus — identical windows for every build.

    .venv/bin/python scripts/ppl_large.py <MODEL_DIR> [CORPUS.npy] [SEQ_LEN] [RESULTS.json]

Per-window NLL goes into RESULTS.json so ppl_compare.py can run the paired
bootstrap across builds (same window set for every entry).
"""
import json
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deepseek_v41_mlx.load import load

MODEL = sys.argv[1]
CORPUS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "ppl_corpus.npy")
SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "..", "ppl_results.json")
try:
    mx.set_wired_limit(int(470e9))
except Exception as e:  # noqa: BLE001
    print("[warn]", e, flush=True)


def bootstrap_ci(win_nll, win_tok, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(win_nll)
    idx = rng.integers(0, k, size=(n, k))
    ppl = np.exp(win_nll[idx].sum(1) / win_tok[idx].sum(1))
    return float(np.percentile(ppl, 2.5)), float(np.percentile(ppl, 97.5))


def main():
    name = os.path.basename(MODEL.rstrip("/"))
    ids_all = np.load(CORPUS)
    n_win = len(ids_all) // SEQ
    print(f"[ppl] {name}: {n_win} windows x {SEQ} tokens = {n_win * SEQ:,}", flush=True)
    model, _ = load(MODEL)  # auto lazy/materialize (see load.py)
    win_nll, win_tok, t0 = [], [], time.time()
    # The window forward runs in chunks (identical numerics — the parity suite
    # proves chunked prefill == single forward): one 2048-token graph over a
    # near-RAM-sized lazy build can stall a Metal command buffer past the GPU
    # watchdog, and after ONE timeout the process's further GPU submissions are
    # ignored (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored) — so
    # prevention is the only strategy that works in-process.
    CHUNK = int(os.environ.get("DSV41_PPL_CHUNK", "256"))

    def window_nll(ids):
        cache = model.make_cache(bsz=1, max_seq_len=SEQ + 8, dtype=mx.bfloat16)
        ids_mx = mx.array([ids])
        pieces = []
        for a in range(0, len(ids), CHUNK):
            lg = model(ids_mx[:, a:a + CHUNK], cache)[0].astype(mx.float32)
            mx.eval(lg)
            pieces.append(lg)
        lg = mx.concatenate(pieces, axis=0)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nll = -lg[mx.arange(len(ids) - 1), mx.array(ids[1:])]
        return float(nll.sum().item())

    for w in range(n_win):
        ids = ids_all[w * SEQ:(w + 1) * SEQ].tolist()
        try:
            total = window_nll(ids)
        except RuntimeError as err:
            if "Timeout" in str(err) or "Ignored" in str(err):
                print(f"[ppl] window {w}: GPU watchdog tripped and further "
                      f"submissions are ignored — rerun (state is per-window); "
                      f"consider a smaller DSV41_PPL_CHUNK", flush=True)
            raise
        win_nll.append(total)
        win_tok.append(len(ids) - 1)
        mx.clear_cache()
        if (w + 1) % 10 == 0 or w == n_win - 1:
            done = sum(win_tok)
            el = time.time() - t0
            print(f"[ppl] {w + 1}/{n_win}  ppl {math.exp(sum(win_nll) / done):.4f}  "
                  f"({done / el:.0f} tok/s, {el:.0f}s, peak {mx.get_peak_memory() / 1e9:.0f} GB)",
                  flush=True)
    win_nll, win_tok = np.array(win_nll), np.array(win_tok)
    ppl = float(np.exp(win_nll.sum() / win_tok.sum()))
    lo, hi = bootstrap_ci(win_nll, win_tok)
    print(f"[ppl] {name}: perplexity {ppl:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"over {int(win_tok.sum()):,} tokens", flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[name] = {"perplexity": round(ppl, 4), "ci95": [round(lo, 4), round(hi, 4)],
                 "tokens": int(win_tok.sum()), "windows": int(n_win), "seq_len": SEQ,
                 "window_nll": win_nll.tolist(), "window_tok": win_tok.tolist()}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[ppl] saved -> {OUT}")


if __name__ == "__main__":
    main()
