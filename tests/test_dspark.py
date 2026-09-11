"""DSpark (MTP) drafter parity vs the reference ``forward_spec`` / ``forward_head`` on the tiny
config: two draft stages transplanted with the same random weights into the reference torch model
(docs/reference/model.py, temperature 0) and into deepseek_v41_mlx.dspark. After a 48-token
prefill seeds both rings, every teacher-forced decode step compares the block logits (Markov
bias applied), the drafted tokens and the confidence scores.

    .venv/bin/python tests/test_dspark.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import test_parity as TP                      # noqa: E402  (reference adaptations, CFG, transplant)
import mlx.core as mx                         # noqa: E402
from deepseek_v41_mlx import dspark as DS      # noqa: E402
from deepseek_v41_mlx.config import ModelArgs  # noqa: E402
from deepseek_v41_mlx.model import Model       # noqa: E402

REF, CFG, VOCAB, S, DECODE, MAXSEQ = TP.REF, TP.CFG, TP.VOCAB, TP.S, TP.DECODE, TP.MAXSEQ
N_MTP, BLOCK = 2, 3
DSPARK = dict(num_nextn_predict_layers=N_MTP, dspark_block_size=BLOCK, dspark_noise_token_id=VOCAB - 1,
              dspark_target_layer_ids=[5, 6, 7], dspark_markov_rank=8,
              dspark_n_routed_experts=4, dspark_num_experts_per_tok=2)


def transplant_mtp(rm, dr, rng):
    D = CFG["dim"]
    def w(*shape, s=None):
        fan = shape[-1] if len(shape) > 1 else shape[0]
        return (rng.standard_normal(shape) * (s if s is not None else fan ** -0.5)).astype(np.float32)
    def setp(param, arr):
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.asarray(arr)))
    def setl(rl, ml, W): setp(rl.weight, W); ml.weight = mx.array(W)
    def setnorm(rn, mn, dim):
        g = (1.0 + 0.1 * rng.standard_normal(dim)).astype(np.float32); setp(rn.weight, g); mn.weight = mx.array(g)
    for si in range(N_MTP):
        rb, mb = rm.mtp[si], dr.stages[si]
        for nm in ("attn", "ffn"):
            fn = w((2 + 4) * 4, 4 * D, s=0.05); ba = (0.2 * rng.standard_normal((2 + 4) * 4)).astype(np.float32)
            sc = (0.5 + 0.2 * rng.standard_normal(3)).astype(np.float32)
            setp(getattr(rb, f"hc_{nm}_fn"), fn); setattr(mb, f"hc_{nm}_fn", mx.array(fn))
            setp(getattr(rb, f"hc_{nm}_base"), ba); setattr(mb, f"hc_{nm}_base", mx.array(ba))
            setp(getattr(rb, f"hc_{nm}_scale"), sc); setattr(mb, f"hc_{nm}_scale", mx.array(sc))
        setnorm(rb.attn_norm, mb.attn_norm, D); setnorm(rb.ffn_norm, mb.ffn_norm, D)
        ra, ma = rb.attn, mb.attn
        NH, HD, QLR, OLR, G = CFG["n_heads"], CFG["head_dim"], CFG["q_lora_rank"], CFG["o_lora_rank"], CFG["o_groups"]
        setl(ra.wq_a, ma.wq_a, w(QLR, D)); setl(ra.wq_b, ma.wq_b, w(NH * HD, QLR)); setl(ra.wkv, ma.wkv, w(HD, D))
        setl(ra.wo_a, ma.wo_a, w(G * OLR, NH * HD // G)); setl(ra.wo_b, ma.wo_b, w(D, G * OLR))
        setnorm(ra.q_norm, ma.q_norm, QLR); setnorm(ra.kv_norm, ma.kv_norm, HD)
        sk = w(NH, s=0.5); setp(ra.attn_sink, sk); ma.attn_sink = mx.array(sk)
        NE, INTER = DSPARK["dspark_n_routed_experts"], CFG["moe_inter_dim"]
        gw = w(NE, D); setp(rb.ffn.gate.weight, gw); mb.ffn.gate.weight = mx.array(gw)
        gb = (0.3 * rng.standard_normal(NE)).astype(np.float32); setp(rb.ffn.gate.bias, gb); mb.ffn.gate.bias = mx.array(gb)
        mb.ffn.gate.bias_vl = mx.zeros((NE,), dtype=mx.float32)
        e1, e2, e3 = w(NE, INTER, D), w(NE, D, INTER), w(NE, INTER, D)
        for i in range(NE):
            setp(rb.ffn.experts[i].w1.weight, e1[i]); setp(rb.ffn.experts[i].w2.weight, e2[i]); setp(rb.ffn.experts[i].w3.weight, e3[i])
        mb.ffn.experts.gate_proj.weight = mx.array(e1); mb.ffn.experts.down_proj.weight = mx.array(e2); mb.ffn.experts.up_proj.weight = mx.array(e3)
        setl(rb.ffn.shared_experts.w1, mb.ffn.shared_experts.w1, w(INTER, D))
        setl(rb.ffn.shared_experts.w2, mb.ffn.shared_experts.w2, w(D, INTER))
        setl(rb.ffn.shared_experts.w3, mb.ffn.shared_experts.w3, w(INTER, D))
        if si == 0:
            setl(rb.main_proj, mb.main_proj, w(D, D * len(DSPARK["dspark_target_layer_ids"]))); setnorm(rb.main_norm, mb.main_norm, D)
        if si == N_MTP - 1:
            setnorm(rb.norm, mb.norm, D)
            R = DSPARK["dspark_markov_rank"]
            setl(rb.markov_head.embed, mb.markov_head.embed, w(VOCAB, R, s=0.5))
            setl(rb.markov_head.head, mb.markov_head.head, w(VOCAB, R, s=0.3))
            cw = w(1, D + R); setp(rb.confidence_head.proj.weight, cw); mb.confidence_head.proj.weight = mx.array(cw)


def main():
    rng = np.random.default_rng(23)
    cfg = dict(CFG); cfg["compress_ratios"] = tuple(CFG["compress_ratios"]) + (0,) * N_MTP   # mtp layers: window only
    rargs = REF.ModelArgs(max_batch_size=1, max_seq_len=MAXSEQ, dtype="bf16", expert_dtype=None, vision_n_layers=0,
                          engram_pad_id=2, temperature=0, n_mtp_layers=N_MTP, dspark_block_size=BLOCK,
                          dspark_noise_token_id=DSPARK["dspark_noise_token_id"],
                          dspark_target_layer_ids=tuple(DSPARK["dspark_target_layer_ids"]),
                          dspark_markov_rank=DSPARK["dspark_markov_rank"],
                          dspark_n_routed_experts=DSPARK["dspark_n_routed_experts"],
                          dspark_n_activated_experts=DSPARK["dspark_num_experts_per_tok"], **cfg)
    margs = ModelArgs(max_seq_len=MAXSEQ, engram_pad_id=2, **CFG)
    with torch.no_grad():
        rm = REF.Transformer(rargs, tokenizer=None).float()
    mm = Model(margs, token_map=list(range(VOCAB)))
    TP.transplant(rm, mm, rng)
    dr = DS.DSpark(margs, DSPARK).bind(mm)
    transplant_mtp(rm, dr, rng)
    for st in dr.stages:
        st.attn.wo_a.weight = st.attn.wo_a.weight.astype(mx.float32)
    mx.eval(mm.parameters(), dr.parameters())
    ids = rng.integers(3, VOCAB - 1, size=(1, S + DECODE + 1)).astype(np.int64)

    # --- prefill: the target's hiddens seed both rings ---
    with torch.no_grad():
        _, _, mh_ref = rm(torch.from_numpy(ids[:, :S]), 0)
        rm.forward_spec(torch.from_numpy(ids[:, S]), mh_ref, 0)
    cache = mm.make_cache(bsz=1, max_seq_len=MAXSEQ, dtype=mx.float32)
    mm._dspark_targets = dr.targets; dr.reset(mx.float32)
    lg, mh = DS.forward_capture(mm, mx.array(ids[:, :S]), cache); mx.eval(lg)
    d = TP.rel_max_diff(np.array(mh), mh_ref.numpy())
    print(f"  target hidden (concat of layers {DSPARK['dspark_target_layer_ids']}) vs reference   {d:.3e}")
    assert d < 2e-5, d
    dr.seed(mh, 0)

    # --- teacher-forced decode steps: compare block logits, tokens, confidence ---
    worst = 0.0; agree = 0; n = 0
    for i in range(S, S + DECODE):
        with torch.no_grad():
            _, _, mh_i = rm(torch.from_numpy(ids[:, i:i + 1]), i)
            out_ref, lg_ref, conf_ref = rm.forward_spec(torch.from_numpy(ids[:, i + 1]), mh_i, i)
        lg, mh = DS.forward_capture(mm, mx.array(ids[:, i:i + 1]), cache); mx.eval(lg)
        dr.seed(mh, i)
        lg_ours, toks, conf = dr.draft_logits(int(ids[0, i + 1]), with_confidence=True)
        mx.eval(lg_ours, conf)
        d = TP.rel_max_diff(np.array(lg_ours), lg_ref.numpy()); worst = max(worst, d)
        dc = TP.rel_max_diff(np.array(conf), conf_ref.numpy()); worst = max(worst, dc)
        agree += int(toks == out_ref[0, 1:].tolist()); n += 1
        print(f"  step {i}: block logits {d:.3e}  confidence {dc:.3e}  tokens ours {toks} ref {out_ref[0, 1:].tolist()}")
    print(f"  worst rel diff {worst:.3e}; drafted blocks identical to the reference: {agree}/{n}")
    assert worst < 2e-5, worst
    assert agree == n
    print("DSPARK PARITY PASS")


if __name__ == "__main__":
    main()
