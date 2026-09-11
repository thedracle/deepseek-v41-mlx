import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32", "gate_f32"]); tok = load_tokenizer(M)
text = open(os.path.expanduser("~/qwen/LOCAL-DEEPSEEK-SETUP.md")).read() * 3
ids = tok(text)["input_ids"][:16384]; print(f"[pc] {len(ids)} tokens", flush=True)
for chunk in (2048, 4096, 8192):
    cache = model.make_cache(bsz=1, max_seq_len=len(ids) + 64, dtype=mx.bfloat16)
    t = time.perf_counter()
    for a in range(0, len(ids), chunk): _forward(model, mx.array([ids[a:a + chunk]]), cache)
    el = time.perf_counter() - t
    print(f"[pc] chunk {chunk}: {el:.2f}s = {len(ids)/el:.0f} tok/s, peak {mx.get_peak_memory()/1e9:.0f} GB", flush=True)
