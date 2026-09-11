"""One load, layered post-load patches: async+wo_a_f32 baseline -> +freqs_pre -> +gate_f32 -> +shared_fuse."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply(VP.DEFAULT_STACK.split(",") + ["async"])
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx import generate as G
from deepseek_v41_mlx.generate import load_tokenizer
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
t0 = time.time(); model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32"]); tok = load_tokenizer(M)
print(f"[bench3] load {time.time()-t0:.1f}s", flush=True)
ids = tok("Explain, step by step, how a copy-on-write B-tree keeps a reader consistent during a page split.")["input_ids"]
def run(n):
    t = time.perf_counter(); out = G.greedy_generate(model, ids, max_new_tokens=n, eos_id=-1, dtype=mx.bfloat16)
    return time.perf_counter() - t, len(out), out
def bench(label, ref=None):
    run(8); t8, n8, _ = run(8); t128, n128, out = run(128)
    same = "" if ref is None else (" | IDENTICAL" if out == ref else " | DIFFERS")
    print(f"[bench3] {label:<28} decode {(n128-n8)/(t128-t8):.2f} tok/s | overall {n128/t128:.2f}{same}", flush=True)
    return out
ref = bench("async+wo_a_f32 (prev best)")
VP.apply_post_load(model, ["freqs_pre"]);   bench("+freqs_pre", ref)
VP.apply_post_load(model, ["gate_f32"]);    bench("+gate_f32", ref)
VP.apply_post_load(model, ["shared_fuse"]); bench("+shared_fuse", ref)
# prefill throughput on a ~1.5k-token prompt
long_ids = (ids * 40)[:1536]
t = time.perf_counter(); G.greedy_generate(model, long_ids, max_new_tokens=1, eos_id=-1, dtype=mx.bfloat16); tp = time.perf_counter() - t
print(f"[bench3] prefill {len(long_ids)} tokens in {tp:.2f}s = {len(long_ids)/tp:.0f} tok/s", flush=True)
