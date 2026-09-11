"""S2: confidence-trimmed drafts. Same prompts as bench_fast, DSpark with conf_min in
(None, 0.0, 0.5, 1.0): tok/s, tokens/step, drafted tokens trimmed.
    python scripts/s2_conf_trim.py <MODEL_DIR>"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer
from deepseek_v41_mlx import dspark as DS
M = sys.argv[1]; model, _ = load(M); tok = load_tokenizer(M); dr = DS.load_dspark(model, M)
PROMPTS = {"prose": "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split. Cover page versioning, root swapping and garbage collection.",
           "code": "Write a Python class LRUCache with get/put in O(1) using an OrderedDict, with docstrings and a small self-test under __main__."}
def chat(p): return tok("<｜begin▁of▁sentence｜><｜User｜>" + p + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
for name, p in PROMPTS.items():
    ids = chat(p)
    for cm in (None, -0.5, 0.0, 0.5, 1.0):
        DS.spec_generate(model, dr, ids[:8], max_new_tokens=6, eos_id=-1, conf_min=cm)
        t = time.perf_counter(); out = DS.spec_generate(model, dr, ids, max_new_tokens=200, eos_id=1, conf_min=cm); dt = time.perf_counter() - t
        s = DS.STATS
        print(f"[s2] {name:5s} conf_min={str(cm):5s} {len(out):4d} tok {dt:5.1f}s = {len(out)/dt:5.1f} tok/s | {len(out)/max(s['steps'],1):.2f} tok/step, accepted {s['accepted']}/{s['drafted']}, trimmed {s['trimmed']}", flush=True)
