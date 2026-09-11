import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
#!/usr/bin/env python3
"""Run DeepSeek-V4.1-Flash through the PipeNetwork port with the optimised default stack.

    python scripts/turbo/run.py "your prompt" [max_new_tokens]

Env: V41_PATCH overrides the patch list (default = v41_patches.DEFAULT_STACK); V41_POST post-load patches;
     V41_SPEC=0 disables the DSpark speculative decoder (needs ~/models/DeepSeek-V4.1-Flash-MLX-dspark, +15 GB);
     DSV41_LAZY=0 is forced (eager load; requires ~427 GB free -> stop oMLX first).
Measured 2026-09-11 on the M3 Ultra 512 GB: 6.69 -> 10.1 tok/s decode with the default stack + post-load patches (V41_POST).
"""
import os, sys, time
os.environ.setdefault("DSV41_LAZY", "0")

import mlx.core as mx
from deepseek_v41_mlx.turbo import patches as v41_patches
stack = os.environ.get("V41_PATCH", v41_patches.DEFAULT_STACK)
v41_patches.apply(stack.split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer

M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain how a copy-on-write B-tree keeps a reader consistent during a page split."
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200

t0 = time.time(); model = load(M); model = model[0] if isinstance(model, tuple) else model
v41_patches.apply_post_load(model, os.environ.get("V41_POST", v41_patches.DEFAULT_POST).split(","))
tok = load_tokenizer(M)
print(f"[run] loaded in {time.time()-t0:.0f}s, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
ids = tok("<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
if os.environ.get("V41_SPEC", "1") == "1":          # native DSpark/MTP speculative decoding (default on)
    from deepseek_v41_mlx.turbo import dspark as D
    drafter = D.load_dspark(model, M)
    t1 = time.time(); out = D.spec_generate(model, drafter, ids, max_new_tokens=n); el = time.time() - t1
    s = D.STATS; print(f"[run] dspark: {s['steps']} steps, {s['accepted']}/{s['drafted']} drafted tokens accepted, {len(out)/max(s['steps'],1):.2f} tok/step", flush=True)
else:
    t1 = time.time(); out = greedy_generate(model, ids, max_new_tokens=n, dtype=mx.bfloat16); el = time.time() - t1
print(f"[run] {len(out)} tokens in {el:.1f}s = {len(out)/el:.2f} tok/s (incl. prefill of {len(ids)})\n")
print(tok.decode(out))
