"""Load target (default stack + wo_a_f32) and the DSpark drafter; compare greedy vs speculative."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply(VP.DEFAULT_STACK.split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx import generate as G
from deepseek_v41_mlx.generate import load_tokenizer
from deepseek_v41_mlx.turbo import dspark as D
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
t0 = time.time(); model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32"]); tok = load_tokenizer(M)
print(f"[dspark] target loaded {time.time()-t0:.0f}s peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
t0 = time.time(); drafter = D.load_dspark(model, M)
print(f"[dspark] drafter loaded {time.time()-t0:.0f}s peak {mx.get_peak_memory()/1e9:.0f} GB active {mx.get_active_memory()/1e9:.0f} GB", flush=True)
PROMPTS = {
 "prose": "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split. Cover page versioning, root swapping and garbage collection.",
 "code": ("Here is a Python function:\n\ndef parse_config(path):\n    with open(path) as f:\n        data = json.load(f)\n"
          "    host = data.get('host', 'localhost')\n    port = data.get('port', 8080)\n    timeout = data.get('timeout', 30)\n"
          "    retries = data.get('retries', 3)\n    return Config(host=host, port=port, timeout=timeout, retries=retries)\n\n"
          "Rewrite it so every field is read with a helper `_get(data, key, default)` that logs a warning when the default is used. "
          "Output only the full rewritten function."),
}
N = int(os.environ.get("N", "160"))
def chat(p):   # no chat_template in the release tokenizer; DeepSeek's user/assistant markers
    return tok("<｜begin▁of▁sentence｜><｜User｜>" + p + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
for name, p in PROMPTS.items():
    ids = chat(p); ids = list(ids["input_ids"]) if isinstance(ids, dict) else list(ids)
    G.greedy_generate(model, ids[:8], max_new_tokens=4, eos_id=-1, dtype=mx.bfloat16)      # warm
    t = time.perf_counter(); base = G.greedy_generate(model, ids, max_new_tokens=N, eos_id=1, dtype=mx.bfloat16); tb = time.perf_counter() - t
    print(f"[dspark] {name}: greedy   {len(base)} tok in {tb:.1f}s = {len(base)/tb:.2f} tok/s", flush=True)
    D.spec_generate(model, drafter, ids[:8], max_new_tokens=6, eos_id=-1)                   # warm
    t = time.perf_counter(); spec = D.spec_generate(model, drafter, ids, max_new_tokens=N, eos_id=1); ts = time.perf_counter() - t
    s = D.STATS; common = 0
    while common < min(len(base), len(spec)) and base[common] == spec[common]: common += 1
    print(f"[dspark] {name}: DSpark   {len(spec)} tok in {ts:.1f}s = {len(spec)/ts:.2f} tok/s | steps {s['steps']} drafted {s['drafted']} "
          f"accepted {s['accepted']} ({100*s['accepted']/max(s['drafted'],1):.0f}%) rejects {s['rejects']} resyncs {s['resyncs']} -> "
          f"{len(spec)/max(s['steps'],1):.2f} tok/step | matches greedy for first {common}/{min(len(base),len(spec))} tokens", flush=True)
    print("[dspark]   greedy:", repr(tok.decode(base)[:300]), flush=True)
    print("[dspark]   dspark:", repr(tok.decode(spec)[:300]), flush=True)
