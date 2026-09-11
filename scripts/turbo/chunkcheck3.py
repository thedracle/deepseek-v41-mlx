"""Which path is right? Compare single-token decode, chunked continuation, and a from-scratch
full prefill (start_pos=0, the path validated against the reference) on the same token sequence."""
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
tok = load_tokenizer(M)
ids = tok("Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split.")["input_ids"]
N = 40; L = len(ids)
cache = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16)
lg = _forward(model, mx.array([ids]), cache); snap0 = _cache_snapshot(cache)
seq, logA = [], []; t = int(mx.argmax(lg[:, -1], axis=-1)[0])
for i in range(N):
    seq.append(t); lg = model(mx.array([[t]]), cache, last_logit_only=True); mx.eval(lg)
    logA.append(lg[0, -1].astype(mx.float32)); t = int(mx.argmax(lg[0, -1]))
    if t == 1: break
N = len(seq)
def stats(name, logs):
    d = [float(mx.max(mx.abs(logs[i] - logA[i]))) for i in range(N)]; s = sorted(d)
    fl = sum(int(mx.argmax(logs[i])) != int(mx.argmax(logA[i])) for i in range(N))
    print(f"[chk3] {name:<34} vs single-token: max {s[-1]:.3f} median {s[N//2]:.3f} flips {fl}/{N}", flush=True)
    return d
# full prefill from scratch of prompt + seq (one chunk, start_pos=0)
c2 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.bfloat16)
lgF = model(mx.array([ids + seq]), c2, last_logit_only=False); mx.eval(lgF)
logF = [lgF[0, L + i].astype(mx.float32) for i in range(N)]
dF = stats("full prefill (start_pos=0)", logF)
# chunked continuation of 4
_cache_restore(cache, snap0); logC = []
for a in range(0, N, 4):
    lg = model(mx.array([seq[a:a + 4]]), cache, last_logit_only=False); mx.eval(lg)
    logC += [lg[0, j].astype(mx.float32) for j in range(lg.shape[1])]
dC = stats("chunked continuation (4)", logC)
d2 = [float(mx.max(mx.abs(logC[i] - logF[i]))) for i in range(N)]; s = sorted(d2)
print(f"[chk3] chunked vs full prefill: max {s[-1]:.3f} median {s[N//2]:.3f}", flush=True)
# fp32 cache dtype: is the gap bf16 cache rounding?
c3 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.float32)
lg = _forward(model, mx.array([ids]), c3); log32 = []; 
for i in range(N):
    lg = model(mx.array([[seq[i]]]), c3, last_logit_only=True); mx.eval(lg); log32.append(lg[0, -1].astype(mx.float32))
stats("single-token, fp32 cache", log32)
c4 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=mx.float32)
lgF = model(mx.array([ids + seq]), c4, last_logit_only=False); mx.eval(lgF)
logF32 = [lgF[0, L + i].astype(mx.float32) for i in range(N)]
d = [float(mx.max(mx.abs(logF32[i] - log32[i]))) for i in range(N)]; s = sorted(d)
print(f"[chk3] fp32 cache: full prefill vs single-token: max {s[-1]:.3f} median {s[N//2]:.3f}", flush=True)
