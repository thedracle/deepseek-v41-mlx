"""Run a fixed number of decode steps for a Metal System Trace (Phase 0 / A1 in
docs/optimization-angles.md). Loads the build, prefills a short chat prompt, warms up, then runs
STEPS greedy decode steps with the fast paths on, each step fully evaluated, separated by a
20 ms sleep so the per-step GPU busy/idle split is readable in the trace.

    xcrun xctrace record --template 'Metal System Trace' --time-limit 240s \
        --output /tmp/v41_decode.trace --launch -- python scripts/trace_decode.py <MODEL_DIR>
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward
MODEL = sys.argv[1]; STEPS = int(os.environ.get("STEPS", "8"))
model, _ = load(MODEL); tok = load_tokenizer(MODEL)
ids = tok("<｜begin▁of▁sentence｜><｜User｜>Explain how a B-tree page split works.<｜Assistant｜>", add_special_tokens=False)["input_ids"]
cache = model.make_cache(bsz=1, max_seq_len=len(ids) + STEPS + 16, dtype=mx.bfloat16)
lg = _forward(model, mx.array([ids]), cache)
t = mx.argmax(lg[:, -1], axis=-1)
for _ in range(3):                                    # warm-up steps (shader compile, first touch)
    lg = model(t[:, None], cache, last_logit_only=True); mx.eval(lg); t = mx.argmax(lg[:, -1], axis=-1); mx.eval(t)
print(f"[trace] warm; tracing {STEPS} decode steps", flush=True)
time.sleep(0.5)
times = []
for i in range(STEPS):
    t0 = time.perf_counter()
    lg = model(t[:, None], cache, last_logit_only=True); t = mx.argmax(lg[:, -1], axis=-1); mx.eval(t)
    times.append(time.perf_counter() - t0)
    time.sleep(0.02)
print("[trace] step ms:", " ".join(f"{x*1000:.0f}" for x in times), flush=True)
time.sleep(0.5)
