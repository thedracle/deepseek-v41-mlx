"""Per-position numerics of the n>1 continuation path vs single-token decode (pre-EOS only)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
stack = os.environ.get("V41_PATCH", VP.DEFAULT_STACK)
if stack: VP.apply(stack.split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward, _cache_snapshot, _cache_restore
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
if "wo_a" in os.environ.get("POST", "wo_a_f32"): VP.apply_post_load(model, ["wo_a_f32"])
tok = load_tokenizer(M)
p = "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split."
try: ids = list(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True))
except Exception: ids = tok(p)["input_ids"]
if isinstance(ids, dict): ids = list(ids["input_ids"])
N = 48
cache = model.make_cache(bsz=1, max_seq_len=len(ids) + N + 16, dtype=mx.bfloat16)
lg = _forward(model, mx.array([ids]), cache); snap0 = _cache_snapshot(cache)
seq, logA = [], []; t = int(mx.argmax(lg[:, -1], axis=-1)[0])
for i in range(N):
    seq.append(t); lg = model(mx.array([[t]]), cache, last_logit_only=True); mx.eval(lg)
    logA.append(lg[0, -1].astype(mx.float32)); t = int(mx.argmax(lg[0, -1]))
    if t == 1: break
N = len(seq); print(f"[chk2] stack={stack!r} {N} greedy tokens (no EOS inside): {tok.decode(seq)[:120]!r}")
def run(label, chunks):
    _cache_restore(cache, snap0); pos = 0; diffs = []; flips = 0
    while pos < N:
        c = min(chunks[len(diffs) % len(chunks)] if isinstance(chunks, list) else chunks, N - pos)
        lg = model(mx.array([seq[pos:pos + c]]), cache, last_logit_only=False); mx.eval(lg)
        for j in range(c):
            d = float(mx.max(mx.abs(lg[0, j].astype(mx.float32) - logA[pos + j]))); diffs.append(d)
            flips += int(int(mx.argmax(lg[0, j])) != int(mx.argmax(logA[pos + j])))
        pos += c
    srt = sorted(diffs)
    print(f"[chk2] {label:<22} max|dlogit| {srt[-1]:.3f}  median {srt[len(srt)//2]:.3f}  p90 {srt[int(len(srt)*.9)]:.3f}  flips {flips}/{N}")
run("n=1 last_logit=False", 1)
run("chunks of 2", 2); run("chunks of 6", 6); run("chunks 1,3,2,6,5,4,8", [1, 3, 2, 6, 5, 4, 8])
