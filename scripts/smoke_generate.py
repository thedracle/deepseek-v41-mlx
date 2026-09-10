"""Greedy generation through this package — a collapse detector, not a quality measure.

    .venv/bin/python scripts/smoke_generate.py <MODEL_DIR> [MAX_TOKENS]

Uses the shipped chat template from tokenizer_config.json when one exists;
falls back to raw completion otherwise.
"""
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer
from deepseek_v41_mlx.load import load

try:
    mx.set_wired_limit(int(470e9))
except Exception as e:  # noqa: BLE001
    print("[warn]", e, flush=True)

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120

t0 = time.time()
model, args = load(path)   # auto lazy/materialize (see load.py)
tok = load_tokenizer(path)
tc_path = os.path.join(path, "tokenizer_config.json")
template = None
if os.path.exists(tc_path):
    tc = json.load(open(tc_path))
    template = tc.get("chat_template")
    if template:
        tok.chat_template = template
print(f"[smoke] loaded in {time.time() - t0:.0f}s "
      f"(chat template: {'yes' if template else 'no'})", flush=True)

for p in ["The capital of France is",
          "Write a Python function that merges overlapping intervals.",
          "Explain in two sentences why the sky appears blue."]:
    if template:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True)
    else:
        ids = tok(p)["input_ids"]
    t0 = time.time()
    out = greedy_generate(model, ids, max_new_tokens=n, dtype=mx.bfloat16)
    print(f"\n=== {p}\n{tok.decode(out)}\n[{time.time() - t0:.1f}s, "
          f"peak {mx.get_peak_memory() / 1e9:.0f} GB]", flush=True)
