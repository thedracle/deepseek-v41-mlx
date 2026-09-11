"""Perplexity arbiter: NLL of the same natural text scored by three paths."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, math, time
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32", "gate_f32"]); tok = load_tokenizer(M)
text = open(os.path.expanduser("~/qwen/LOCAL-DEEPSEEK-SETUP.md")).read()[3000:9000]
ids = tok(text)["input_ids"][:640]; T = len(ids); P = 64          # prefix P, score positions P..T-1
def nll_from(logits_fn):
    """logits_fn(i) -> fp32 logits predicting ids[i]; returns mean NLL over i in [P, T)."""
    tot = 0.0
    for i in range(P, T):
        lg = logits_fn(i); lp = lg - mx.logsumexp(lg); tot += -float(lp[ids[i]])
    return tot / (T - P)
def score(label, chunk):
    cache = model.make_cache(bsz=1, max_seq_len=T + 8, dtype=mx.bfloat16)
    logits = {}
    t0 = time.time()
    lg = _forward(model, mx.array([ids[:P]]), cache); logits[P] = lg[0, -1].astype(mx.float32)
    pos = P
    while pos < T - 1:
        n = min(chunk, T - 1 - pos)
        lg = model(mx.array([ids[pos:pos + n]]), cache, last_logit_only=False); mx.eval(lg)
        for j in range(n): logits[pos + j + 1] = lg[0, j].astype(mx.float32)
        pos += n
    nll = nll_from(lambda i: logits[i])
    print(f"[chk5] {label:<36} NLL {nll:.4f}  ppl {math.exp(nll):.2f}  ({time.time()-t0:.0f}s)", flush=True)
score("1-token decode after 64-token prefix", 1)
score("8-token chunks", 8)
score("64-token chunks", 64)
score("one chunk (full continuation)", T)
# full prefill from position 0
cache = model.make_cache(bsz=1, max_seq_len=T + 8, dtype=mx.bfloat16)
lg = model(mx.array([ids]), cache, last_logit_only=False); mx.eval(lg)
nll = nll_from(lambda i: lg[0, i - 1].astype(mx.float32))
print(f"[chk5] {'full prefill from 0':<36} NLL {nll:.4f}  ppl {math.exp(nll):.2f}", flush=True)
