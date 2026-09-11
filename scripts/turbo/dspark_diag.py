"""Drafter quality in isolation: run plain greedy (n=1 path), and at every step ask the DSpark
drafter for a block from the current token; score each draft position against the tokens greedy
actually produced next. Also time draft() and the verify-sized forward."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward
from deepseek_v41_mlx.turbo import dspark as D
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32", "gate_f32"]); tok = load_tokenizer(M)
drafter = D.load_dspark(model, M)
import deepseek_v41_mlx.hyper_connections as HC
print(f"[diag] drafter hc_mixes is {D.hc_mixes.__module__}.{D.hc_mixes.__name__}; HC.hc_mixes is {HC.hc_mixes.__name__}", flush=True)
p = "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split."
ids = tok("<｜begin▁of▁sentence｜><｜User｜>" + p + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
N = 120; L = len(ids); model._dspark_targets = drafter.targets
cache = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16); drafter.reset()
lg, mh = D.forward_capture(model, mx.array([ids]), cache); mx.eval(lg); drafter.seed(mh, 0)
t = int(mx.argmax(lg[:, -1], axis=-1)[0]); seq = [t]; drafts = []; td = 0.0; tv = 0.0
for i in range(N):
    t0 = time.perf_counter(); d = drafter.draft(seq[-1]); td += time.perf_counter() - t0; drafts.append(d)
    lg, mh = D.forward_capture(model, mx.array([[seq[-1]]]), cache); mx.eval(lg)
    drafter.seed(mh, cache.offset - 1)
    seq.append(int(mx.argmax(lg[0, -1])))
    if seq[-1] == 1: break
B = drafter.block_size; pos_hit = [0] * B; pos_n = [0] * B; prefix = [0] * (B + 1); steps = 0
for i, d in enumerate(drafts):
    truth = seq[i + 1:i + 1 + B]
    if len(truth) < B: break
    steps += 1; k = 0
    for j in range(B):
        pos_n[j] += 1; hit = d[j] == truth[j]; pos_hit[j] += hit
        if k == j and hit: k += 1
    prefix[k] += 1
print(f"[diag] {steps} steps; per-position draft accuracy: " + " ".join(f"p{j}={100*pos_hit[j]/max(pos_n[j],1):.0f}%" for j in range(B)), flush=True)
print(f"[diag] accepted-prefix length histogram (0..{B}): {prefix}  -> mean tokens/step (prefix+1) = {sum((k+1)*c for k,c in enumerate(prefix))/max(steps,1):.2f}", flush=True)
print(f"[diag] draft() {1000*td/len(drafts):.1f} ms/call; n=1 target forward+seed incl. in loop", flush=True)
# timing of verify-sized forwards
for n in (1, 6, 12):
    x = mx.array([seq[:n]]); snap = cache.offset
    t0 = time.perf_counter()
    for _ in range(5):
        lg, mh = D.forward_capture(model, x, cache); mx.eval(lg, mh); cache.offset = snap
    print(f"[diag] forward_capture n={n}: {1000*(time.perf_counter()-t0)/5:.1f} ms", flush=True)
print("[diag] greedy text:", repr(tok.decode(seq)[:200]), flush=True)
print("[diag] sample drafts vs truth:", [(tok.decode(drafts[i]), tok.decode(seq[i+1:i+1+B])) for i in (0, 5, 20, 40)], flush=True)
