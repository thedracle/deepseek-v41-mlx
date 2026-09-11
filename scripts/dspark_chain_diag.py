"""S1 feasibility: can a SECOND draft block be chained before verifying? The drafter's ring holds
keys made from TARGET hiddens, which do not exist for tokens that have not been verified — so a
chained block must either (a) seed the ring with a stand-in (the drafter's own final residual for
the first block's positions) or (b) skip those keys and just sit at later positions.
Teacher-forced greedy decode; per-position accuracy of block 1 (positions 1-5) and of the chained
block 2 (positions 6-10) under both variants, against the tokens greedy actually produced.

    python scripts/dspark_chain_diag.py <MODEL_DIR> [steps]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer
from deepseek_v41_mlx import dspark as D
M = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
model, _ = load(M); tok = load_tokenizer(M); dr = D.load_dspark(model, M); B = dr.block_size
PROMPTS = {"prose": "Write a detailed, multi-paragraph explanation of how a copy-on-write B-tree keeps a reader consistent during a page split.",
           "code": "Write a Python class LRUCache with get/put in O(1) using an OrderedDict, with docstrings and a small self-test under __main__."}
def chat(p): return tok("<｜begin▁of▁sentence｜><｜User｜>" + p + "<｜Assistant｜>", add_special_tokens=False)["input_ids"]
for name, p in PROMPTS.items():
    ids = chat(p); L = len(ids); model._dspark_targets = dr.targets
    cache = model.make_cache(bsz=1, max_seq_len=L + N + 32, dtype=mx.bfloat16); dr.reset(mx.bfloat16)
    lg, mh = D.forward_capture(model, mx.array([ids]), cache); mx.eval(lg); dr.seed(mh, 0)
    t = int(mx.argmax(lg[:, -1], axis=-1)[0]); seq = [t]; drafts = {"b1": [], "a": [], "b": []}; conf1 = []; tdraft = 0.0
    for i in range(N):
        t0 = time.perf_counter()
        rings = [r[:] for r in dr.rings]; lp = dr.last_pos
        lg1, t1, c1, xp = dr.draft_logits(seq[-1], with_confidence=True, return_hidden=True)
        mx.eval(xp); drafts["b1"].append(t1); conf1.append(c1)
        # (a) stand-in keys: seed every stage's ring with the drafter's final residual for the block positions
        cos, sin = dr._freqs(lp + 2 * B + 4)
        for st, r in zip(dr.stages, dr.rings): st.attn.seed(xp, lp + 1, cos, sin, r)
        dr.last_pos = lp + B
        _, t2a, _ = dr.draft_logits(t1[-1], with_confidence=True); drafts["a"].append(t2a)
        dr.rings = rings; dr.last_pos = lp                       # roll back
        # (b) no keys for the block-1 tokens: block 2 at positions lp+6..lp+10 over the same ring
        _, t2b, _ = dr.draft_logits(t1[-1], with_confidence=True, pos_offset=B); drafts["b"].append(t2b)
        tdraft += time.perf_counter() - t0
        lg, mh = D.forward_capture(model, mx.array([[seq[-1]]]), cache); mx.eval(lg); dr.seed(mh, cache.offset - 1)
        seq.append(int(mx.argmax(lg[0, -1])))
        if seq[-1] == 1: break
    steps = len(drafts["b1"]); hit = {k: [0] * B for k in drafts}; n = [0] * B
    for i in range(steps):
        truth = seq[i + 1:i + 1 + 2 * B]
        if len(truth) < 2 * B: break
        for j in range(B):
            n[j] += 1
            hit["b1"][j] += drafts["b1"][i][j] == truth[j]
            hit["a"][j] += drafts["a"][i][j] == truth[B + j]
            hit["b"][j] += drafts["b"][i][j] == truth[B + j]
    # block-2 accuracy conditional on block 1 being fully right (the only case where it is used)
    cond = {"a": [0] * B, "b": [0] * B}; cn = 0
    for i in range(steps):
        truth = seq[i + 1:i + 1 + 2 * B]
        if len(truth) < 2 * B or drafts["b1"][i] != truth[:B]: continue
        cn += 1
        for j in range(B):
            cond["a"][j] += drafts["a"][i][j] == truth[B + j]; cond["b"][j] += drafts["b"][i][j] == truth[B + j]
    fmt = lambda h, m: " ".join(f"{100*h[j]/max(m,1):3.0f}%" for j in range(B))
    print(f"[chain] {name}: {steps} steps, draft+2 chains {1000*tdraft/steps:.1f} ms/step", flush=True)
    print(f"[chain]   block 1 positions 1-5 accuracy:          {fmt(hit['b1'], n[0])}", flush=True)
    print(f"[chain]   block 2 (a: stand-in keys) positions 6-10: {fmt(hit['a'], n[0])}   | given block 1 all-correct ({cn} steps): {fmt(cond['a'], cn)}", flush=True)
    print(f"[chain]   block 2 (b: no keys)       positions 6-10: {fmt(hit['b'], n[0])}   | given block 1 all-correct: {fmt(cond['b'], cn)}", flush=True)
    c = mx.concatenate(conf1, axis=0); print(f"[chain]   confidence head on block 1: mean {float(c.mean()):.3f}, per position {' '.join(f'{float(v):.2f}' for v in c.mean(axis=0))}", flush=True)
