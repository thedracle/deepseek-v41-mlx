"""DeepSeek-V4.1-Flash native DSpark (MTP) speculative decoding for the PipeNetwork MLX port.

Reference: inference/model.py (DSparkAttention / DSparkBlock / forward_spec) in deepseek-ai/DeepSeek-V4.1-Flash.
Three draft stages (mtp.0..2) each run one HC block whose attention reads a private sliding-window
ring of keys derived from `main_x` = main_norm(main_proj(concat of the target's hc-mean hidden
entering layers 37,38,39)); the drafted block ([bonus, noise x4]) supplies the queries and a
transient block KV (non-causal within the block). Stage 2 ends with norm -> target head (+ a low-rank
Markov bias conditioned on the previous drafted token). The reference ships only the forward;
the verify loop here is standard greedy block verification on the target's chunk-general forward
with snapshot/rollback on rejection (see spec_generate)."""
import dataclasses, json, os, time
import mlx.core as mx, mlx.nn as nn
from deepseek_v41_mlx.layers import RMSNorm, rope_tail
from deepseek_v41_mlx.fakequant import fake_quant_fp8_ue8m0
from deepseek_v41_mlx.moe import MoE
from deepseek_v41_mlx.hyper_connections import hc_mixes, hc_pre, hc_post, make_identity_pre_mix
from deepseek_v41_mlx.model import SharedState
from deepseek_v41_mlx.generate import _cache_snapshot, _cache_restore
from deepseek_v41_mlx.load import quant_predicate

DST = os.path.expanduser("~/models/DeepSeek-V4.1-Flash-MLX-dspark")


def attn_full_sink(q, kv, sink, scale):
    """q [b,B,h,d] attends to ALL of kv [b,n,d] (one shared K=V vector per position, MLA-style)
    plus the per-head sink logit (a zero-value key), via mx.fast SDPA (GQA path)."""
    b, B, h, d = q.shape; n = kv.shape[1]
    kvx = mx.concatenate([kv, mx.zeros((b, 1, d), dtype=kv.dtype)], axis=1)[:, None].astype(mx.float32)
    mask = mx.concatenate([mx.zeros((b, h, B, n), dtype=mx.float32),
                           mx.broadcast_to(sink.astype(mx.float32).reshape(1, h, 1, 1), (b, h, B, 1))], axis=-1)
    o = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3).astype(mx.float32), kvx, kvx, scale=scale, mask=mask)
    return o.transpose(0, 2, 1, 3).astype(q.dtype)


class DSparkAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim, self.n_heads, self.head_dim, self.rd = args.dim, args.n_heads, args.head_dim, args.rope_head_dim
        self.n_groups, self.o_lora_rank, self.window, self.eps = args.o_groups, args.o_lora_rank, args.window_size, args.norm_eps
        self.softmax_scale = args.head_dim ** -0.5
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(self.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

    def seed(self, main_x, pos0, cos, sin, ring):
        """Write the ring keys for positions pos0..pos0+n-1 (slot = pos % window)."""
        n = main_x.shape[1]
        kv = self.kv_norm(self.wkv(main_x))
        kv = rope_tail(kv, self.rd, cos[pos0:pos0 + n], sin[pos0:pos0 + n])
        kv = fake_quant_fp8_ue8m0(kv, 32)
        if n > self.window:
            kv = kv[:, -self.window:]; pos0 += n - self.window; n = self.window
        slots = (pos0 + mx.arange(n)) % self.window
        ring[:, slots] = kv.astype(ring.dtype)
        return ring

    def __call__(self, x, last_pos, cos, sin, ring):
        """x [1,B,dim] = the draft block at positions last_pos+1..last_pos+B."""
        b, B, _ = x.shape
        c, s = cos[last_pos + 1:last_pos + 1 + B], sin[last_pos + 1:last_pos + 1 + B]
        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(b, B, self.n_heads, self.head_dim)
        q = rope_tail(q, self.rd, c, s)
        kv = fake_quant_fp8_ue8m0(rope_tail(self.kv_norm(self.wkv(x)), self.rd, c, s), 32)
        k0 = min(self.window, last_pos + 1)
        kv_all = mx.concatenate([ring[:, :k0].astype(kv.dtype), kv], axis=1)
        o = attn_full_sink(q, kv_all, self.attn_sink, self.softmax_scale)
        o = rope_tail(o, self.rd, c, s, inverse=True)
        o = o.reshape(b, B, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        o = mx.einsum("bsgd,grd->bsgr", o.astype(mx.float32), wo_a.astype(mx.float32))
        return self.wo_b(o.reshape(b, B, -1).astype(x.dtype))


class MarkovHead(nn.Module):
    def __init__(self, vocab, rank):
        super().__init__()
        self.embed = nn.Embedding(vocab, rank)
        self.head = nn.Linear(rank, vocab, bias=False)


class ConfidenceHead(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.proj = nn.Linear(dim_in, 1, bias=False)


class DSparkStage(nn.Module):
    def __init__(self, args, cfg, stage_id, n_stages):
        super().__init__()
        self.norm_eps, self.hc_mult, self.hc_iters, self.hc_eps = args.norm_eps, args.hc_mult, args.hc_sinkhorn_iters, args.hc_eps
        margs = dataclasses.replace(args, n_routed_experts=cfg["dspark_n_routed_experts"],
                                    n_activated_experts=cfg["dspark_num_experts_per_tok"])
        self.attn = DSparkAttention(args)
        self.ffn = MoE(margs)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc, hc_dim = (2 + args.hc_mult) * args.hc_mult, args.hc_mult * args.dim
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32); self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32); self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32); self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)
        if stage_id == 0:
            self.main_proj = nn.Linear(args.dim * len(cfg["dspark_target_layer_ids"]), args.dim, bias=False)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == n_stages - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = MarkovHead(args.vocab_size, cfg["dspark_markov_rank"])
            self.confidence_head = ConfidenceHead(args.dim + cfg["dspark_markov_rank"])

    def __call__(self, x, pre_mix, last_pos, cos, sin, ring):
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                                  self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = self.attn(self.attn_norm(hc_pre(x, pre_mix)), last_pos, cos, sin, ring)
        x = hc_post(h, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                                               self.hc_mult, self.hc_iters, self.norm_eps, self.hc_eps)
        h = self.ffn(self.ffn_norm(hc_pre(x, attn_pre)))
        return hc_post(h, residual, ffn_post, ffn_comb), ffn_pre


class DSpark(nn.Module):
    def __init__(self, args, cfg):
        super().__init__()
        self.args, self.cfg = args, cfg
        self.n_stages = cfg["num_nextn_predict_layers"]
        self.block_size = cfg["dspark_block_size"]; self.noise = cfg["dspark_noise_token_id"]
        self.targets = list(cfg["dspark_target_layer_ids"]); self.window = args.window_size
        self.stages = [DSparkStage(args, cfg, i, self.n_stages) for i in range(self.n_stages)]
        self._embed = self._head = self._freqs = None
        self.rings = None; self.last_pos = -1

    def bind(self, target):
        self._embed, self._head, self._freqs = target.embed, target.head, target.layers[0].attn._freqs
        return self

    def reset(self, dtype=mx.bfloat16):
        self.rings = [mx.zeros((1, self.window, self.args.head_dim), dtype=dtype) for _ in self.stages]
        self.last_pos = -1

    def seed(self, main_hidden, pos0):
        """main_hidden [1,n,3*dim]: the target's hc-mean hidden entering each target layer, for
        positions pos0..pos0+n-1 (all of which are now in the target cache)."""
        n = main_hidden.shape[1]
        cos, sin = self._freqs(pos0 + n + self.block_size + 2)
        st0 = self.stages[0]
        main_x = st0.main_norm(st0.main_proj(main_hidden))
        self.rings = [st.attn.seed(main_x, pos0, cos, sin, r) for st, r in zip(self.stages, self.rings)]
        self.last_pos = pos0 + n - 1

    def draft(self, input_token):
        """Greedy block draft: returns the block_size tokens predicted to follow `input_token`
        (which sits at position last_pos+1 and is not yet in the target cache)."""
        B, lp = self.block_size, self.last_pos
        cos, sin = self._freqs(lp + B + 2)
        ids = mx.array([[input_token] + [self.noise] * (B - 1)])
        x = self._embed(ids)
        x = mx.broadcast_to(x[:, :, None, :], (1, B, self.args.hc_mult, x.shape[-1]))
        pre_mix = make_identity_pre_mix(1, B, self.args.hc_mult)
        for st, ring in zip(self.stages, self.rings):
            x, pre_mix = st(x, pre_mix, lp, cos, sin, ring)
        last = self.stages[-1]
        h = last.norm(hc_pre(x, pre_mix))
        logits = self._head(h.astype(mx.float32))                      # [1, B, V]
        mk = last.markov_head
        prev = mx.array([input_token]); toks = []
        for i in range(B):
            bias = mk.head(mk.embed(prev)).astype(mx.float32)          # [1, V]
            prev = mx.argmax(logits[:, i] + bias, axis=-1)
            toks.append(prev)
        return mx.stack(toks)[:, 0].tolist()


def load_dspark(target, model_dir, path=DST):
    raw = json.load(open(os.path.join(model_dir, "config.json")))
    def find(d, key):                       # the MLX config nests the text fields
        if isinstance(d, dict):
            if key in d: return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None: return r
        return None
    cfg = {k: find(raw, k) for k in ("num_nextn_predict_layers", "dspark_block_size", "dspark_noise_token_id",
                                     "dspark_target_layer_ids", "dspark_markov_rank", "dspark_n_routed_experts",
                                     "dspark_num_experts_per_tok")}
    assert None not in cfg.values(), cfg
    d = DSpark(target.args, cfg)
    qm = json.load(open(os.path.join(path, "quantization.json")))
    nn.quantize(d, group_size=qm["group_size"], bits=qm["bits"],
                class_predicate=quant_predicate(qm["group_size"], qm["bits"], None, module_map=qm["modules"]))
    w = mx.load(os.path.join(path, "dspark.safetensors"))
    d.load_weights(list(w.items()), strict=True)
    d.stages[-1].confidence_head.proj.weight = d.stages[-1].confidence_head.proj.weight.astype(mx.float32)
    for st in d.stages:                                                 # same trick as wo_a_f32 on the target
        st.attn.wo_a.weight = st.attn.wo_a.weight.astype(mx.float32)
    mx.eval(d.parameters())
    return d.bind(target)


def forward_capture(model, input_ids, cache):
    """The port's Model.__call__ plus capture of the DSpark target hiddens (the hc-mean of the
    stream ENTERING each target layer, after that layer's engram). Returns (logits [1,n,V] fp32,
    main_hidden [1,n,len(targets)*dim])."""
    import numpy as np
    start_pos = cache.offset; b, n = input_ids.shape
    hashes = None
    if model.engram_hasher is not None:
        hashes = mx.array(model.engram_hasher(np.array(input_ids, dtype=np.int64), start_pos, cache.engram_ids))
    h = model.embed(input_ids)
    h = mx.broadcast_to(h[:, :, None, :], (b, n, model.hc_mult, h.shape[-1]))
    pre_mix = make_identity_pre_mix(b, n, model.hc_mult); shared = SharedState(); mains = []
    targets = set(model._dspark_targets)
    for i, layer in enumerate(model.layers):
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
        if i in targets:
            mains.append(mx.mean(h.astype(mx.float32), axis=2).astype(h.dtype))
        h, pre_mix = layer(h, pre_mix, start_pos, cache, shared)
    h = model.norm(hc_pre(h, pre_mix))
    logits = model.head(h.astype(mx.float32))
    cache.offset = start_pos + n
    return logits, mx.concatenate(mains, axis=-1)


STATS = {}


def spec_generate(model, drafter, input_ids, max_new_tokens=64, max_seq_len=None, eos_id=1,
                  dtype=mx.bfloat16, prefill_chunk=512, max_pending=12):
    """Greedy speculative decoding with the native DSpark drafter. Each step: draft block_size
    tokens from the last decided token, forward [pending + draft] through the target in one
    chunk, accept the longest matching prefix + 1 bonus token from the target's own logits.
    Rejection rolls the target cache back (snapshot) and carries the accepted tokens as
    `pending` into the next forward (bounded by max_pending, beyond which they are re-forwarded
    once). The drafter's rings are seeded with the target hiddens of every accepted token."""
    try: mx.set_wired_limit(int(470e9))
    except Exception: pass
    for k in ("steps", "drafted", "accepted", "rejects", "resyncs"): STATS[k] = 0
    model._dspark_targets = drafter.targets
    seq = list(input_ids) if not hasattr(input_ids[0], "__len__") else list(input_ids[0])
    B = drafter.block_size
    cache = model.make_cache(bsz=1, max_seq_len=max_seq_len or (len(seq) + max_new_tokens + max_pending + B + 8), dtype=dtype)
    drafter.reset(dtype)
    ids = mx.array([seq]); logits = None
    for a in range(0, len(seq), prefill_chunk):
        logits, mh = forward_capture(model, ids[:, a:a + prefill_chunk], cache)
        mx.eval(logits); drafter.seed(mh, a)
    first = int(mx.argmax(logits[:, -1], axis=-1)[0])
    out, pending = [first], [first]
    while len(out) < max_new_tokens and out[-1] != eos_id:
        p0 = cache.offset
        draft = drafter.draft(pending[-1])
        inp = pending + draft
        STATS["steps"] += 1; STATS["drafted"] += len(draft)
        snap = _cache_snapshot(cache)
        lg, mh = forward_capture(model, mx.array([inp]), cache)
        preds = mx.argmax(lg[0], axis=-1).tolist()
        P = len(pending); acc = 0
        for j, d in enumerate(draft):
            if preds[P - 1 + j] == d: acc += 1
            else: break
        STATS["accepted"] += acc
        bonus = preds[P - 1 + acc]
        drafter.seed(mh[:, :P + acc], p0)
        out += draft[:acc] + [bonus]
        if acc == len(draft):
            pending = [bonus]
        else:
            STATS["rejects"] += 1
            _cache_restore(cache, snap)
            pending = pending + draft[:acc] + [bonus]
            if len(pending) > max_pending:
                STATS["resyncs"] += 1
                forward_capture(model, mx.array([pending[:-1]]), cache)
                pending = [bonus]
    if eos_id in out: out = out[:out.index(eos_id)]
    return out[:max_new_tokens]
