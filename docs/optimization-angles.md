# DeepSeek-V4.1-Flash on M3 Ultra — remaining optimization angles

Baseline (2026-09-11, fork `thedracle/deepseek-v41-mlx` branch `docs`, unpruned mixed-4/8 build):
greedy 9.8 tok/s (105 ms/step, ~6,500 Metal launches), DSpark 13 tok/s prose / 22 code,
agent tool-call turns ~30, prefill 379 tok/s (2048-token chunks). Perplexity 2.8972 (paired
ratio 1.0000 vs the untouched port). Everything below is measured against this.

Bandwidth floor: ~10 GB of weights are read per decode step (6 experts × 35 M params at 4-bit
≈ 4.2 GB, attention/shared/gate at 8-bit ≈ 5 GB) → ~13 ms at 800 GB/s. Decode runs 8× above
its floor. The gap is launch/latency overhead, which bounds how much fusion can still give.

Rules learned the hard way (see docs/fast-and-dspark.md "what lost"):
- benchmark with real shapes AND real index patterns; random-index micro-benches lie
- an in-situ A/B through the server is the acceptance test, not a kernel micro-bench
- one hand-written GEMV kernel cannot fill the GPU (64 heads = 64 threadgroups); fuse chains, not matmuls
- quality gate for anything that changes numerics: paired perplexity on identical windows

## Phase 0 — analysis (do first; turns estimates into measurements)

| # | angle | what it tells us | effort |
|---|---|---|---|
| A1 | Metal GPU trace of one decode step (`mx.metal.start_capture` → Instruments) | GPU busy vs idle fraction; launch histogram by kernel name; count of `copy`/`astype`/`concatenate` kernels; whether Python dispatch starves the queue | 1–2 h |
| A2 | Sync-point audit | every `.item()/int()/tolist()/np.array()` per step. Known suspect: the engram hasher runs in numpy every token (`Model.__call__` → `np.array(input_ids)`), forcing a GPU drain before the next graph can be built — likely why `mx.async_eval` gained only +1.3 % | 1 h |
| A3 | Per-layer launch and time split | window-only vs compressed-KV layers; indexer/compressor share at decode | 1 h |
| A4 | MLX version bump (0.32.2 → current) | free kernel/compile improvements (small-M qmm, SDPA, compile with updates); must precede any kernel work or every benchmark is against a moving target | 1 h + rerun of the three test suites |

## Phase 1 — speculative decoding (verifier unchanged → no quality risk)

| # | angle | estimate | effort |
|---|---|---|---|
| S1 | Confidence-chained drafting: when the loaded confidence head is high on the tail, draft a second block from the last drafted token before verifying (8–10 speculative tokens per verify; verify cost is nearly flat: n=6 146 ms, n=12 179 ms) | code +30–50 % (22 → 30+), agent turns 30 → 40+, prose +5 % | afternoon |
| S2 | Skip/shorten drafts when confidence is low | prose +3–5 % (fewer wasted verify tokens) | hours |
| S3 | Draft trees: branch top-2 at the first uncertain position, tree-masked verify | prose 2.5 → ~3.5 tok/step (+30 %), code +10 % | 2–3 days |
| S4 | Rejection sampling for temperature > 0 | 0 % faster; removes the greedy-only limit | afternoon |

## Phase 2 — launch count (greedy and every DSpark step)

| # | angle | estimate | effort |
|---|---|---|---|
| L1 | Remove the per-token CPU sync (engram hash on-device in MLX ops, or hashed one step behind) so `async_eval` actually pipelines | +10–20 % greedy if A2 confirms the drain | half day |
| L2 | `mx.compile` per block after making cache updates functional (return new arrays instead of in-place writes) | +20–40 % greedy if the compiler cooperates; ceiling from the bandwidth floor is far higher | 1–2 days, real risk |
| L3 | Hand-fuse remaining chains: q-proj + q_norm, attn-out → wo_a → wo_b, gate → argpartition; eliminate `copy` kernels A1 finds | +5–10 % | 1–2 days |

## Phase 3 — kernels (only with A1/A4 in hand and a candidate list from upstream research)

| # | angle | estimate | effort / risk |
|---|---|---|---|
| K1 | Fused decode MoE: route + gather + SwiGLU + weighted sum in one kernel (MoE ≈ 12 ms of the 105) | ≤ 8 % per step | 3–5 days, medium risk of losing to `gather_qmm` |
| K2 | Expert-major prefill MoE: stream each expert's weights once per chunk, tokens sorted by expert (prefill is 62 % MoE and weight-bandwidth-bound) | prefill 1.5–2× (379 → 550–750) | ~1 week, high risk, largest prize |
| K3 | Small-M quantized matmul premium (213 → 280 µs at M=1→6; ~40 ms of every verify chunk) | −25 % verify cost if fixed in MLX | upstream issue/PR search first |
| K4 | Prefill sparse attention with `simdgroup_matrix` tiles (oMLX `wsdpa` structure) | prefill +10–15 % | 3–4 days; previous attempt lost; after K2 only |

Research sources for the candidate list: MLX release notes and `gather_qmm`/SDPA/compile source; mlx-lm's
MoE path; oMLX `wsdpa`; llama.cpp Metal MoE; DeepSeek `inference/kernel.py` (TileLang DSA kernels);
FlashMLA Metal ports. Candidate bar: ≥1.5× on real shapes and localized index lists in a micro-bench,
then survives the in-situ A/B.

## Not worth pursuing (measured or bounded)

- pruning for speed: REAP25 +12 % greedy, +1 % with DSpark, ×1.028 ppl — memory tool, not speed
- decode attention kernels: already one SDPA call per layer (< 2 %)
- bigger prefill chunks on the unpruned build: +1–8 % but eats the ~20 GB headroom
- REAP37/50: quality visibly degrades

## Plan of record

1. Phase 0 in one GPU session (A4 first, then A1–A3 on the bumped MLX).
2. S1 immediately after — highest ratio of gain to effort, zero quality exposure.
3. L1 if A2 confirms the hasher drain; then L3; L2 as a bounded experiment.
4. Phase 3 only from a written candidate list, one kernel at a time, in-situ A/B before merge.

Expected landing if 1–3 go as estimated: code/agent turns 35–45 tok/s, greedy prose 11–13, prefill unchanged.
Quality gate on every step: three test suites + paired perplexity where numerics change.
