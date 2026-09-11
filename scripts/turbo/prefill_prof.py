"""Prefill profile: where do the ms go for a 512/2048-token chunk? Class-level wrappers with forced eval."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # repo root
import os, sys, time, collections
import mlx.core as mx

from deepseek_v41_mlx.turbo import patches as VP
VP.apply("sinkhorn_metal,hc_metal,rmsnorm,idxcache,fq_metal,rope_metal,sa_sdpa".split(","))
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import load_tokenizer, _forward
from deepseek_v41_mlx import attention as A, moe as MOE, engram as E, layers as L, compressor as CP, indexer as IX, sparse_attention as SA, hyper_connections as HC
M = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit")
model = load(M); model = model[0] if isinstance(model, tuple) else model
VP.apply_post_load(model, ["wo_a_f32", "gate_f32"]); tok = load_tokenizer(M)
text = open(os.path.expanduser("~/qwen/LOCAL-DEEPSEEK-SETUP.md")).read()
ids = tok(text)["input_ids"][:4096]
# uninstrumented timings first
for chunk in (512, 1024, 2048):
    cache = model.make_cache(bsz=1, max_seq_len=len(ids) + 64, dtype=mx.bfloat16)
    t = time.perf_counter()
    for a in range(0, len(ids), chunk): _forward(model, mx.array([ids[a:a + chunk]]), cache)
    el = time.perf_counter() - t
    print(f"[pp] prefill {len(ids)} tokens, chunk {chunk}: {el:.2f}s = {len(ids)/el:.0f} tok/s", flush=True)
T = collections.defaultdict(float); C = collections.Counter()
def wrap(cls, name):
    orig = cls.__call__
    def w(self, *a, **k):
        t = time.perf_counter(); out = orig(self, *a, **k)
        mx.eval(out) if isinstance(out, mx.array) else mx.eval([o for o in (out if isinstance(out, tuple) else ()) if isinstance(o, mx.array)])
        T[name] += time.perf_counter() - t; C[name] += 1; return out
    cls.__call__ = w
def wrapfn(mod, fname, name):
    orig = getattr(mod, fname)
    def w(*a, **k):
        t = time.perf_counter(); out = orig(*a, **k); mx.eval(out) if isinstance(out, mx.array) else mx.eval(list(out))
        T[name] += time.perf_counter() - t; C[name] += 1; return out
    setattr(mod, fname, w)
wrap(A.Attention, "attn (whole)"); wrap(MOE.MoE, "ffn/moe (whole)"); wrap(E.Engram, "engram (whole)")
wrap(L.RMSNorm, "  rmsnorm"); wrap(CP.Compressor, "  compressor"); wrap(IX.Indexer, "  indexer"); wrap(MOE.Gate, "  moe.gate"); wrap(MOE.SharedExpert, "  moe.shared")
wrapfn(SA, "sparse_attn", "  sparse_attn"); wrapfn(HC, "hc_mixes", "  hc_mixes"); wrapfn(HC, "hc_post", "  hc_post")
if hasattr(A, "sparse_attn"): A.sparse_attn = SA.sparse_attn
hasher = model.engram_hasher
if hasher is not None:
    ho = type(hasher).__call__
    def hw(self, *a, **k):
        t = time.perf_counter(); r = ho(self, *a, **k); T["  hasher(numpy,cpu)"] += time.perf_counter()-t; C["  hasher(numpy,cpu)"] += 1; return r
    type(hasher).__call__ = hw
chunk = 512; ids2 = ids[:2048]
cache = model.make_cache(bsz=1, max_seq_len=len(ids2) + 64, dtype=mx.bfloat16)
t = time.perf_counter()
for a in range(0, len(ids2), chunk): _forward(model, mx.array([ids2[a:a + chunk]]), cache)
tot = time.perf_counter() - t; n = len(ids2) // chunk
print(f"[pp] instrumented: {len(ids2)} tokens in {chunk}-chunks: {tot:.2f}s ({tot/n*1000:.0f} ms/chunk)\n", flush=True)
print(f"  {'component':<24}{'ms/chunk':>10}{'calls/chunk':>12}{'ms/call':>9}{'share':>8}"); print("  " + "-" * 63)
for k, v in sorted(T.items(), key=lambda x: (x[0].startswith("  "), -x[1])):
    print(f"  {k:<24}{v/n*1000:>10.1f}{C[k]/n:>12.0f}{v*1000/max(C[k],1):>9.3f}{v/tot*100:>7.1f}%", flush=True)
top = sum(v for k, v in T.items() if not k.startswith("  "))
print(f"  {'(glue)':<24}{(tot-top)/n*1000:>10.1f}{'':>12}{'':>9}{(tot-top)/tot*100:>7.1f}%", flush=True)
