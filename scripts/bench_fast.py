"""Decode / prefill figures for a converted build: reference path (DSV41_FAST=0), fast paths,
and DSpark speculative decoding if <model>/dspark exists. Chat-formatted prompts, EOS honoured.

    python scripts/bench_fast.py <MODEL_DIR> [--tokens 160]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx import fast
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer, _forward
from deepseek_v41_mlx import dspark as DS

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--tokens", type=int, default=160); a = ap.parse_args()
t0 = time.time(); model, _ = load(a.model); tok = load_tokenizer(a.model)
print(f"[bench] {os.path.basename(a.model.rstrip('/'))}: loaded in {time.time()-t0:.0f}s, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
PROMPTS = {
    "prose": "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split. Cover page versioning, root swapping and garbage collection.",
    "code": ("Here is a Python function:\n\ndef parse_config(path):\n    with open(path) as f:\n        data = json.load(f)\n"
             "    host = data.get('host', 'localhost')\n    port = data.get('port', 8080)\n    timeout = data.get('timeout', 30)\n"
             "    retries = data.get('retries', 3)\n    return Config(host=host, port=port, timeout=timeout, retries=retries)\n\n"
             "Rewrite it so every field is read with a helper `_get(data, key, default)` that logs a warning when the default is used. "
             "Output only the full rewritten function."),
}
def chat(p): return tok("<｜begin▁of▁sentence｜><｜User｜>" + p + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
def run(fn, ids, n, **kw):
    fn(ids[:8], 4, **kw)                                   # warm
    t = time.perf_counter(); out = fn(ids, n, **kw); return len(out), time.perf_counter() - t, out
def greedy(ids, n): return greedy_generate(model, ids, max_new_tokens=n, eos_id=1, dtype=mx.bfloat16)
drafter = None
if os.path.exists(os.path.join(a.model, "dspark", "dspark.safetensors")):
    drafter = DS.load_dspark(model, a.model); print(f"[bench] dspark head loaded, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
def spec(ids, n): return DS.spec_generate(model, drafter, ids, max_new_tokens=n, eos_id=1)
for name, p in PROMPTS.items():
    ids = chat(p); rows = []
    fast.enable(False); n, dt, ref = run(greedy, ids, a.tokens); rows.append(("reference path", n, dt, ""))
    fast.enable(True);  n, dt, out = run(greedy, ids, a.tokens); rows.append(("fast paths", n, dt, "identical" if out == ref else "DIFFERS"))
    if drafter:
        n, dt, out = run(spec, ids, a.tokens); s = DS.STATS
        rows.append(("fast + dspark", n, dt, f"{s['accepted']}/{s['drafted']} accepted, {n/max(s['steps'],1):.2f} tok/step"))
    for lab, n, dt, note in rows:
        print(f"[bench] {name:5s} {lab:16s} {n:4d} tok {dt:6.1f}s = {n/dt:5.1f} tok/s (incl. {len(ids)}-token prefill)  {note}", flush=True)
text = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read() * 8
ids = tok(text)["input_ids"][:8192]
for chunk in (512, 2048):
    cache = model.make_cache(bsz=1, max_seq_len=len(ids) + 64, dtype=mx.bfloat16)
    t = time.perf_counter()
    for s0 in range(0, len(ids), chunk): _forward(model, mx.array([ids[s0:s0 + chunk]]), cache)
    dt = time.perf_counter() - t
    print(f"[bench] prefill {len(ids)} tokens, chunk {chunk}: {len(ids)/dt:.0f} tok/s, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
