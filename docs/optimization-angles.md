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



### S1 result: chained drafting is not viable (measured, `scripts/dspark_chain_diag.py`)

The ring the drafter attends to is built from TARGET hiddens (`main_proj` of layers 37/38/39), which do
not exist for unverified tokens. A second block drafted before verification, teacher-forced over 150
steps, per-position accuracy for positions 6–10:

| variant | prose | code | given block 1 all-correct |
|---|---|---|---|
| block 1 (positions 1–5, for reference) | 80 61 41 28 16 % | 87 72 57 47 33 % | — |
| (a) ring seeded with the drafter's own residual | 5 1 2 3 2 % | 11 6 7 3 2 % | prose 20 7 0 0 0; code 26 13 5 5 5 |
| (b) no keys for block-1 positions | 4 4 4 2 3 % | 14 10 9 8 8 % | prose 13 0 13 0 0; code 38 26 23 18 13 |

Dead. The confidence head is well calibrated (mean logit per position: prose 2.75 1.21 0.43 −0.18
−0.58; code 4.77 2.89 1.72 1.17 0.60), so S2 (trim the block where confidence < 0) remains, worth a few
percent on prose. The remaining speculative lever is S3 (draft trees within one block: the Markov
sequential argmax can produce a sibling branch from the same block logits at no extra drafter cost;
the verify needs a tree-aware window index matrix — the compressed-KV pollution is the same class as
the chunk path, and the committed state is always re-forwarded clean).

## Phase 0 results (2026-09-11)

- **A4 — MLX bump: nothing available.** 0.32.2 and mlx-lm 0.31.3 are the newest releases.
- **A2 — sync audit: one host round-trip per token, confirmed.** `Model.__call__` → `np.array(input_ids)` for
  the engram hasher. Fixed on branch `hasher` (hash in MLX ops, int64 semantics verified bit-for-bit,
  history as an mx.array; numpy path stays the reference path). All three suites pass.
- **A1 — Metal System Trace of 8 decode steps (`scripts/trace_decode.py`, `metal-gpu-intervals`):**
  - **GPU busy 99 %** of the step: 89.2 ms busy of 90.1 ms wall, ~220 command buffers per step
    (~30 kernels each), idle gaps between command buffers ~1 µs (p90 1 µs), CPU→GPU queue depth
    ~2.8 ms. The GPU is never starved within a step.
  - The bench's 105 ms/step vs the trace's 90 ms GPU time ⇒ ~15 ms/step is CPU graph construction
    that runs *serially* after the previous step's sync. That 15 ms (≈14 %) is the entire prize for
    pipelining (L1 + async_eval); it cannot exceed that.
  - Therefore "launch-bound" means: ~6,500 kernels averaging ~14 µs each, each paying fixed
    per-dispatch cost *inside* the GPU (scheduling, barriers, cache flushes), not a CPU that can't feed
    it. The lever is **fewer, larger kernels** (L2 `mx.compile` fusion, L3 hand-fused chains, K1 fused
    MoE), and the bandwidth floor (~13 ms) says the ceiling for that is large.
  - Per-kernel names/durations need the shader-profiler instrument (empty in the System Trace
    template); the command-buffer view is enough to settle the busy/idle question.
  - **Graph build measured** (tiny config, same op structure, scaled 8 → 40 layers): 7 ms/step with the
    fast paths (16 ms on the reference path). So the CPU side is ~7 % of a step and pipelining could
    never hide more than that — consistent with async_eval (+1.3 %) and the on-device hasher
    (greedy 9.9 vs 9.8 prose, 8.6 vs 8.4 code: noise-level). **L1 is a cleanliness win, not a speed win.**
  - Conclusion for the plan: decode time is ~90 ms of GPU execution of ~6,500 small kernels. Only fusion
    (fewer kernels: L2/L3/K1) or fewer steps (DSpark: S1/S3) can move it. Pipelining is exhausted.

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
