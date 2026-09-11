# Fast paths, DSpark speculative decoding, serving

Measured on a Mac Studio M3 Ultra (512 GB), 2026-09-11, on this branch, one process per build,
same prompts, only the flag flipped. Perplexity through `scripts/ppl_large.py` + `ppl_compare.py`
(wikitext-2 test, 286,580 tokens, 2048-token windows, paired bootstrap over identical windows).

## Decode

| build | reference path | `fast` | `fast` + DSpark |
|---|---:|---:|---:|
| mixed 4/8, engram-4 (427 GB) — prose | 6.6 tok/s | 9.8 | 12.9 (2.76 tok/step, 35 % accept) |
| mixed 4/8, engram-4 — code | 6.4 | 8.4 | **21.9** (4.55 tok/step, 78 %) |
| REAP25 (351 GB) — prose | 7.2 | 11.2 | 13.0 (2.46 tok/step, 29 %) |
| REAP25 — code | 6.8 | 9.5 | **23.9** (4.76 tok/step, 80 %) |

Coding-agent use through `scripts/serve_openai.py` (pi, 19.4k-token system prompt): 28–31 tok/s on
tool-call turns at 95–97 % draft acceptance, 15–18 tok/s on long thinking-mode answers.

## Prefill

| chunk | mixed 4/8 | REAP25 | peak (with the 15 GB drafter loaded) |
|---:|---:|---:|---:|
| 512 (default) | 282 tok/s | 296 | 451 / 374 GB |
| 2048 | 379 | 389 | 453 / 376 GB |
| 4096 · 8192 (unpruned, no drafter) | 382 · 408 | | 442 · 453 GB |

Prefill is MoE weight-bandwidth-bound at small chunks (every chunk re-reads a layer's 6.8 GB of
experts); chunk size is the lever, attention kernels are not (see "what lost" below).

`greedy_generate` and `ppl_large.py` now default to 2048-token chunks. The 256–512 defaults upstream
guard against a *lazy* (mmap) load near the RAM ceiling, where one large chunk can stall a Metal
command buffer on page-ins and a single timeout poisons the process. A materialized build (what
`load()` picks whenever it fits) ran 2048-token chunks for hours, and 8192 for a 16k prompt, without
tripping the watchdog. `DSV41_PPL_CHUNK=256` reproduces the README perplexities bit-for-bit.

## Quality: paired perplexity, same windows

| A vs B | ppl B/A | 95 % CI | B better (windows) |
|---|---:|---:|---:|
| reference path vs `fast` (unpruned) | **1.0000** | [0.9988, 1.0012] | 52.9 % — not significant |
| unpruned vs REAP25 (both `fast`) | 1.0280 | [1.0186, 1.0386] | 22.1 % — significant |

Absolute: reference 2.8972 [2.7101, 3.0941] (README: 2.8963), `fast` 2.8972, REAP25 2.9784 (×1.0280;
README: ×1.0281). The fast path's greedy token sequence diverges from the reference path's within
~100–160 tokens on bf16 (accumulation-order differences flip low-margin argmaxes, amplified through
40 hyper-connection layers); the paired test says that divergence carries no quality signal.

## Validation

- `tests/test_parity.py` with `DSV41_FAST=0`: numbers byte-identical to `main`.
- `tests/test_fast.py`: each kernel vs its pure-MLX function (fake-quant bit-exact incl. e4m3
  midpoints; fp32 kernels ≤ 2e-7; bf16-I/O cases ≤ 1.1e-4), then the full reference battery with
  the fast paths on: every row ≤ 3.7e-6, PARITY PASS.
- `tests/test_dspark.py`: two draft stages transplanted into the reference torch model
  (`forward_spec`/`forward_head`, temperature 0) and into `deepseek_v41_mlx.dspark`; after a
  48-token prefill seeds both rings, 8 teacher-forced steps: block logits, confidence ≤ 1.1e-6,
  all drafted blocks identical.

## Why decode got faster (and what lost)

Decode is launch-bound: ~6,500 Metal kernel launches per token at ~23 µs each, weights are not the
limit. So the wins are fused elementwise/reduction chains — Sinkhorn split (20 sweeps in registers),
hc_pre/hc_post, the three fake-quant formats, the RoPE tail, `mx.fast.rms_norm`, SDPA with the sink
as one zero-value key — plus not casting a 134 MB unquantized `wo_a` to fp32 per layer per token,
and pipelining the decode loop with `mx.async_eval`.

What lost, kept in the history with the reasons: hand-fused GEMV kernels for `hc_mixes`, the MoE gate
and sparse attention (64 heads = 64 threadgroups cannot fill the GPU from one launch; 1.5–8× slower);
fusing the shared-expert w1/w3 GEMVs (different tile, slower); precomputing RoPE tables (no effect);
`sparse_attn4`, a tiled prefill attention kernel that wins a random-index micro-bench 1.2–2.9× and
loses in situ 330 vs 365 tok/s — real window/top-k index lists are localized and the reference
einsum is a batched GEMM. Benchmark attention kernels with real index patterns.

## DSpark in one paragraph

The release ships three draft stages (`mtp.0..2.*`, 14.2 B params, shards 44–46) that `convert.py`
drops. `scripts/convert_mtp.py` builds `<model>/dspark/` from them (8 GB download, 15.4 GB MLX
8-bit; the "I8" experts are packed fp4 pairs like the main model's). Each stage is one HC block whose
attention reads a private 128-slot key ring derived from the target's hidden state entering layers
37/38/39; the draft block `[bonus, noise×4]` is denoised non-causally in one pass, then the target
head plus a Markov bias on the previous drafted token gives five tokens. Verification forwards
`[pending + draft]` through the target in one chunk (n=6 costs 146 ms vs 105 ms for n=1 while
launch-bound), accepts the longest matching prefix plus the target's own next token, and rolls the
cache back on rejection. Greedy only.

## Usage

```
python scripts/convert_mtp.py --model <MODEL_DIR>            # once, builds <MODEL_DIR>/dspark
python scripts/bench_fast.py <MODEL_DIR>                     # the table above
python scripts/serve_openai.py --model <MODEL_DIR> --port 8001
DSV41_FAST=0 ...                                             # pure reference path everywhere
```
