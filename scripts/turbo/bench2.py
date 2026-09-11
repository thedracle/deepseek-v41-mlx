"""One load, many configs. Loads with DEFAULT_STACK, then layers post-load patches
(async / wo_a_f32 / specdec) and benches each. Decode tok/s = (n128-n8)/(t128-t8), EOS off."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply(VP.DEFAULT_STACK.split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx import generate as G
from deepseek_v41_mlx.generate import load_tokenizer
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
t0 = time.time(); model = load(M); model = model[0] if isinstance(model, tuple) else model
tok = load_tokenizer(M)
print(f"[bench2] load {time.time()-t0:.1f}s peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
PROSE = "Explain, step by step, how a copy-on-write B-tree keeps a reader consistent during a page split."
CODE = ("Here is a Python function:\n\ndef parse_config(path):\n    with open(path) as f:\n        data = json.load(f)\n"
        "    host = data.get('host', 'localhost')\n    port = data.get('port', 8080)\n    timeout = data.get('timeout', 30)\n"
        "    retries = data.get('retries', 3)\n    return Config(host=host, port=port, timeout=timeout, retries=retries)\n\n"
        "Rewrite it so every field is read with a helper `_get(data, key, default)` that logs a warning when the default is used. "
        "Output only the full rewritten function.")
def run(prompt, n):
    ids = tok(prompt)["input_ids"]
    t = time.perf_counter(); out = G.greedy_generate(model, ids, max_new_tokens=n, eos_id=-1, dtype=mx.bfloat16)
    return time.perf_counter() - t, len(out), out
def bench(label, prompt=PROSE, ref=None):
    run(prompt, 8)
    t8, n8, _ = run(prompt, 8); t128, n128, out = run(prompt, 128)
    dec = (n128 - n8) / (t128 - t8)
    extra = ""
    if hasattr(G, "SPEC_STATS"):
        s = G.SPEC_STATS; extra = f" | spec steps {s['steps']} drafted {s['drafted']} accepted {s['accepted']} rejects {s['rejects']} nodraft {s['nodraft']} -> {n128/max(s['steps'],1):.2f} tok/step"
    same = "" if ref is None else (" | output IDENTICAL to baseline" if out[:n128] == ref[:n128] else " | output DIFFERS from baseline")
    print(f"[bench2] {label:<26} decode {dec:.2f} tok/s | overall {n128/t128:.2f} | {n128} tok {t128:.1f}s{extra}{same}", flush=True)
    return out
base_prose = bench("baseline (default stack)")
base_code = bench("baseline code-prompt", CODE)
print("[bench2] +", VP.PATCHES["async"](), flush=True);      bench("+async", ref=base_prose)
print("[bench2] +", VP.PATCHES["wo_a_f32"](), flush=True);   VP.apply_post_load(model, ["wo_a_f32"]); bench("+async+wo_a_f32", ref=base_prose)
print("[bench2] +", VP.PATCHES["specdec"](), flush=True);    bench("specdec prose", ref=base_prose); bench("specdec code-prompt", CODE, ref=base_code)
os.environ["V41_SPEC_K"] = "4"; VP.PATCHES["specdec"]();     bench("specdec K=4 code-prompt", CODE, ref=base_code)
print("[bench2] output tail:", tok.decode(base_code)[-200:].replace("\n", "\\n"), flush=True)
