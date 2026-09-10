"""Strict-load a built checkpoint through this package, then generate.

``deepseek_v41_mlx.load.load`` is strict in BOTH directions — every model
parameter must come from the checkpoint and every checkpoint tensor must land
somewhere (vision passthrough is declared, engram token map rebuilt from the
bundled tokenizer). This is the check that must pass before anything is
published. No mlx-lm, no bundled model_file: builds load through this package
only.

    .venv/bin/python scripts/check_strict_load.py <MODEL_DIR> [MAX_TOKENS]
"""
import sys
import time

import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer
from deepseek_v41_mlx.load import load

try:
    mx.set_wired_limit(int(470e9))
except Exception as e:  # noqa: BLE001
    print("[warn]", e, flush=True)

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100

t0 = time.time()
model, args = load(path, strict=True)  # auto lazy/materialize (see load.py)
print(f"[strict] loaded in {time.time() - t0:.0f}s: zero missing / zero "
      f"unexpected tensors; vision passthrough: {len(model._vision_passthrough)}",
      flush=True)
tok = load_tokenizer(path)
for p in ["The capital of France is",
          "Write a Python function that merges overlapping intervals.",
          "Explain in two sentences why the sky appears blue."]:
    ids = tok(p)["input_ids"]
    t0 = time.time()
    out = greedy_generate(model, ids, max_new_tokens=n, dtype=mx.bfloat16)
    print(f"\n=== {p}\n{tok.decode(out)}\n[{time.time() - t0:.1f}s, "
          f"peak {mx.get_peak_memory() / 1e9:.0f} GB]", flush=True)
