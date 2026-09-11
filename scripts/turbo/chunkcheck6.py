"""Is the chunk-vs-single gap bf16 kernel-order noise or a semantic difference? Run the whole
model with fp32 activations (cast every bf16 parameter to fp32; quantized matmuls output x.dtype)
and repeat the comparison."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys
import mlx.core as mx
from mlx.utils import tree_map

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward, _cache_snapshot, _cache_restore
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
tok = load_tokenizer(M)
ids = tok("<｜begin▁of▁sentence｜><｜User｜>Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split.<｜Assistant｜>", add_special_tokens=False)["input_ids"]
def compare(label, N=24, cdt=mx.bfloat16):
    L = len(ids)
    cache = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=cdt)
    lg = _forward(model, mx.array([ids]), cache); snap0 = _cache_snapshot(cache)
    seq, logA = [], []; t = int(mx.argmax(lg[:, -1], axis=-1)[0])
    for i in range(N):
        seq.append(t); lg = model(mx.array([[t]]), cache, last_logit_only=True); mx.eval(lg)
        logA.append(lg[0, -1].astype(mx.float32)); t = int(mx.argmax(lg[0, -1]))
        if t == 1: break
    N = len(seq)
    def rep(name, pairs):
        d = sorted(float(mx.max(mx.abs(a - b))) for a, b in pairs); fl = sum(int(mx.argmax(a)) != int(mx.argmax(b)) for a, b in pairs)
        print(f"[chk6] {label} {name:<28} n={len(d):2d} max {d[-1]:.4f} median {d[len(d)//2]:.4f} flips {fl}", flush=True)
    _cache_restore(cache, snap0); pairs = []
    for a in range(0, N, 4):
        lg = model(mx.array([seq[a:a + 4]]), cache, last_logit_only=False); mx.eval(lg)
        pairs += [(lg[0, j].astype(mx.float32), logA[a + j]) for j in range(lg.shape[1])]
    rep("chunks of 4 vs single", pairs)
    c2 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=cdt)
    lgF = model(mx.array([ids + seq]), c2, last_logit_only=False); mx.eval(lgF)
    rep("full prefill vs single", [(lgF[0, L + i].astype(mx.float32), logA[i]) for i in range(N)])
    # same path twice: determinism check
    c3 = model.make_cache(bsz=1, max_seq_len=L + N + 16, dtype=cdt)
    lgG = model(mx.array([ids + seq]), c3, last_logit_only=False); mx.eval(lgG)
    rep("full prefill run-to-run", [(lgF[0, L + i].astype(mx.float32), lgG[0, L + i].astype(mx.float32)) for i in range(N)])
compare("bf16 acts:")
n = [0]
def up(a):
    if isinstance(a, mx.array) and a.dtype == mx.bfloat16: n[0] += 1; return a.astype(mx.float32)
    return a
model.update(tree_map(up, model.parameters())); mx.eval(model.parameters())
print(f"[chk6] cast {n[0]} bf16 parameter arrays to fp32", flush=True)
compare("fp32 acts:", cdt=mx.float32)
