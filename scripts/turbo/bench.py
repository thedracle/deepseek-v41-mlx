"""Correct decode benchmark. Warmup first, EOS disabled so every run yields exactly
max_new_tokens, tok/s uses the ACTUAL generated count. V41_PATCH=sinkhorn applies patches."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

if os.environ.get("V41_PATCH"):
    from deepseek_v41_mlx.turbo import patches as v41_patches; v41_patches.apply(os.environ["V41_PATCH"].split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
t0 = time.time(); model = load(M)
if isinstance(model, tuple): model = model[0]
tok = load_tokenizer(M)
print(f"[bench] load {time.time()-t0:.1f}s  peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
ids = tok("Explain, step by step, how a copy-on-write B-tree keeps a reader consistent during a page split.")["input_ids"]
def run(n):
    t = time.perf_counter(); out = greedy_generate(model, ids, max_new_tokens=n, eos_id=-1, dtype=mx.bfloat16)
    return time.perf_counter() - t, len(out), out
run(8)                                   # warmup: shader compile + first touch
t8, n8, _ = run(8); t128, n128, out = run(128)
dec = (n128 - n8) / (t128 - t8)
print(f"[bench] {n8} tok {t8:.2f}s | {n128} tok {t128:.2f}s | decode {dec:.2f} tok/s | overall {n128/t128:.2f} tok/s | peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
print("[bench] output:", tok.decode(out)[:160].replace("\n"," "), flush=True)
