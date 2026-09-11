"""Are only NON-final chunk positions inconsistent (within-chunk future visibility), or every position?"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward, _cache_snapshot, _cache_restore
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
tok = load_tokenizer(M)
ids = tok("<｜begin▁of▁sentence｜><｜User｜>Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split.<｜Assistant｜>", add_special_tokens=False)["input_ids"]
N = 48; L = len(ids)
cache = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16)
lg = _forward(model, mx.array([ids]), cache); snap0 = _cache_snapshot(cache)
seq, logA = [], []; t = int(mx.argmax(lg[:, -1], axis=-1)[0])
for i in range(N):
    seq.append(t); lg = model(mx.array([[t]]), cache, last_logit_only=True); mx.eval(lg)
    logA.append(lg[0, -1].astype(mx.float32)); t = int(mx.argmax(lg[0, -1]))
    if t == 1: break
N = len(seq); print(f"[chk4] {N} tokens: {tok.decode(seq)[:100]!r}", flush=True)
def report(name, pairs):
    if not pairs: return
    d = sorted(float(mx.max(mx.abs(a - b))) for a, b in pairs); fl = sum(int(mx.argmax(a)) != int(mx.argmax(b)) for a, b in pairs)
    print(f"[chk4] {name:<44} n={len(d):3d} max {d[-1]:.3f} median {d[len(d)//2]:.3f} flips {fl}", flush=True)
for C in (2, 4, 8):
    _cache_restore(cache, snap0); last, nonlast = [], []
    for a in range(0, N, C):
        lg = model(mx.array([seq[a:a + C]]), cache, last_logit_only=False); mx.eval(lg); n = lg.shape[1]
        for j in range(n):
            (last if j == n - 1 else nonlast).append((lg[0, j].astype(mx.float32), logA[a + j]))
    report(f"chunks of {C}: LAST position of each chunk", last); report(f"chunks of {C}: non-final positions", nonlast)
# full prefill: only the final position
c2 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16)
lgF = model(mx.array([ids + seq]), c2, last_logit_only=False); mx.eval(lgF)
report("full prefill: final position only", [(lgF[0, L + N - 1].astype(mx.float32), logA[N - 1])])
report("full prefill: all generated positions", [(lgF[0, L + i].astype(mx.float32), logA[i]) for i in range(N)])
# prefill of the prompt in two ways: one chunk vs 8-token chunks -> compare the NEXT-token logits
c3 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16); lg3 = None
for a in range(0, L, 8): lg3 = model(mx.array([ids[a:a + 8]]), c3, last_logit_only=True); mx.eval(lg3)
c4 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16); lg4 = model(mx.array([ids]), c4, last_logit_only=True); mx.eval(lg4)
report("prompt prefill: 8-token chunks vs one chunk (next-token)", [(lg3[0, -1].astype(mx.float32), lg4[0, -1].astype(mx.float32))])
