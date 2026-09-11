"""L2-lite: mx.compile the shape-static FFN half of every block (hc_mixes -> hc_pre -> ffn_norm ->
MoE -> hc_post) at decode and measure greedy tok/s against the uncompiled fast path. The attention
half is left alone (its shapes change with the cache position, which would recompile every step).

    python scripts/compile_experiment.py <MODEL_DIR> [tokens]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
from deepseek_v41_mlx.load import load
from deepseek_v41_mlx.generate import greedy_generate, load_tokenizer
from deepseek_v41_mlx import model as MODEL
from deepseek_v41_mlx.hyper_connections import hc_mixes, hc_pre, hc_post
M = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
model, _ = load(M); tok = load_tokenizer(M)
ids = tok("<｜begin▁of▁sentence｜><｜User｜>Explain, step by step, how a copy-on-write B-tree keeps a reader consistent during a page split.<｜Assistant｜>", add_special_tokens=False)["input_ids"]
def run(label):
    greedy_generate(model, ids[:8], max_new_tokens=4, eos_id=-1, dtype=mx.bfloat16)
    t = time.perf_counter(); o8 = greedy_generate(model, ids, max_new_tokens=8, eos_id=-1, dtype=mx.bfloat16); t8 = time.perf_counter() - t
    t = time.perf_counter(); out = greedy_generate(model, ids, max_new_tokens=N, eos_id=-1, dtype=mx.bfloat16); tN = time.perf_counter() - t
    print(f"[compile] {label:32s} decode {(len(out)-len(o8))/(tN-t8):5.2f} tok/s", flush=True); return out
base = run("fast paths, uncompiled")
# --- compile the FFN half per block ---
orig_call = MODEL.Block.__call__
def make_half(blk):
    def half(x, attn_pre):
        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base, blk.hc_mult, blk.hc_iters, blk.norm_eps, blk.hc_eps)
        h = blk.ffn(blk.ffn_norm(hc_pre(x, attn_pre)))
        return hc_post(h, residual, ffn_post, ffn_comb), ffn_pre
    return mx.compile(half)
for blk in model.layers: blk._ffn_half = make_half(blk)
def call(self, x, pre_mix, start_pos, cache, shared):
    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
    h = self.attn(self.attn_norm(hc_pre(x, pre_mix)), start_pos, cache, shared)
    x = hc_post(h, residual, attn_post, attn_comb)
    return self._ffn_half(x, attn_pre)
MODEL.Block.__call__ = call
t0 = time.perf_counter(); out = run("FFN half compiled (per block)"); print(f"[compile] (first compiled run incl. compile time {time.perf_counter()-t0:.1f}s) output {'identical' if out == base else 'DIFFERS'}", flush=True)
out = run("FFN half compiled, 2nd run"); print(f"[compile] output {'identical' if out == base else 'DIFFERS'}", flush=True)
# --- also compile the attention PRE half (hc_mixes + hc_pre + attn_norm), shape-static ---
def make_pre(blk):
    def pre(x, pre_mix):
        attn_pre, attn_post, attn_comb = hc_mixes(x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base, blk.hc_mult, blk.hc_iters, blk.norm_eps, blk.hc_eps)
        return blk.attn_norm(hc_pre(x, pre_mix)), attn_pre, attn_post, attn_comb
    return mx.compile(pre)
for blk in model.layers: blk._attn_pre = make_pre(blk)
def call2(self, x, pre_mix, start_pos, cache, shared):
    residual = x
    h, attn_pre, attn_post, attn_comb = self._attn_pre(x, pre_mix)
    h = self.attn(h, start_pos, cache, shared)
    x = hc_post(h, residual, attn_post, attn_comb)
    return self._ffn_half(x, attn_pre)
MODEL.Block.__call__ = call2
run("FFN half + attn-pre compiled"); out = run("FFN half + attn-pre, 2nd run"); print(f"[compile] output {'identical' if out == base else 'DIFFERS'}", flush=True)
