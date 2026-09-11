# turbo — DeepSeek-V4.1-Flash on Apple silicon, faster

Working notes, in the order the work happened (M3 Ultra, 512 GB). Numbers are measured, not
projected; losing approaches are kept because they say why.

## DeepSeek-V4.1-Flash optimisation pass (2026-09-11)

Model: `~/models/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit` (427.6 GB, pipenetwork build). Runtime: the
PipeNetwork port at `/tmp/dsv41` (venv: python3.11, mlx 0.32.2, mlx-lm 0.31.3). Not oMLX-servable yet.

### Result
| stack | decode tok/s |
|---|---|
| baseline (eager load, unpatched) | 6.69 |
| + Sinkhorn `mx.compile` | 7.36 |
| + Sinkhorn Metal, hc_pre/hc_post Metal, sync-free loop | 8.01 |
| + `mx.fast.rms_norm`, window-index memoisation | 8.40 |
| + rope_tail Metal, fake-quant Metal x3 | 8.93 (+33%) |
| **+ sparse_attn via SDPA sink-key (DEFAULT STACK)** | **9.02 (+35%)** |

Run it: `python scripts/turbo/run.py "prompt"` (stop oMLX first: needs 427 GB).
Patches live in `scripts/turbo/patches.py` (monkeypatch, port untouched; `DEFAULT_STACK`), kernels in
`scripts/turbo/metal.py`, benchmark `scripts/turbo/bench.py`, profiler `scripts/turbo/profile.py`.

### What the bottleneck actually was
NOT disk streaming (eager 6.69 vs lazy 6.53 — my earlier "126 GB/token" theory was wrong) and NOT the
engram tables (1.2% of decode; the per-token gather is 48 rows). Decode is **launch-bound**: ~6,500 tiny
Metal launches/token at ~23 µs each ≈ the whole 149 ms budget; only ~10 ms is weight bandwidth.
Profile (class-level instrumentation; shares): attention 37%, MoE 31%, block glue 31%, engram 1%,
numpy hasher 0.1%.

### Kernels (all verified vs the port on synthetic tensors)
| kernel | replaces | numerics | in default stack |
|---|---|---|---|
| sinkhorn_split | 20-sweep python loop, 80x/tok | max diff 1.2e-7 | yes |
| hc_pre / hc_post | 4-launch chains, 160x/tok | fp32 exact; bf16 1 ulp | yes |
| rope_tail | ~12 launches, ~136x/tok | **bit-exact** | yes |
| fake_quant fp8_ue8m0 / fp4_ue8m0 / fp4_e4m3 | ~25-50 launches, ~56x/tok | **bit-exact** incl. e4m3/e2m1 ties + subnormals | yes |
| sparse_attn v1/v2/v3 | ~20 launches, 40x/tok | v3: 3e-5 | no — v3 (split-K) 0.711 vs port 0.401 ms: only 64 heads -> 64 threadgroups, GPU under-occupied; einsum spreads the work wider. Would need split-K across threadgroups + a 2nd reduce launch (ceiling ~+15%). Opt-in as `sa_metal`. |
| hc_mixes (fused GEMV) | ~5 launches | fp32-ulp | no — 7x slower |
| moe gate (fused GEMV+topk) | ~8 launches | exact expert sets | no — 8.5x slower (same single-threadgroup-GEMV flaw) |

**Lesson:** fusing elementwise/reduction chains into one launch wins every time; folding a GEMV into a
single-threadgroup hand kernel loses to MLX's matmul every time. `mx.compile` around these chains gave
nothing (already at the launch floor). Instance-level `__call__` monkeypatches are ignored (Python
resolves `__call__` on the type) — patch the class.

### Gotchas
- `half` is a reserved type in Metal; don't name a variable that.
- The port's `greedy_generate` returns ids (not a stream); benchmark decode by differencing a short and
  a long run on the same prompt, after a warmup, with `eos_id=-1` and the ACTUAL token count.
- `DSV41_LAZY=0` on a 512 GB Mac with oMLX stopped: peak 427 GB, compressor 2 GB, zero swap, no stall
  (the author's stall was a 476 GB build).

### Addendum (2026-09-11): what other optimised kernels taught us

**The decode sparse-attention answer is `mx.fast.scaled_dot_product_attention`, not a custom kernel.**
Express the per-head attention sink as ONE extra key with a zero value vector, set its logit to
`attn_sink[h]` via the additive mask, masked indices -> -1e30. Mathematically identical to the port's
denominator-only sink (softmax shift invariance). MLA's single KV head is SDPA's GQA fast path.
- matches the port to 1.2e-4 (bf16 ulp), exact on all-masked rows
- 0.353 ms vs port 0.398 ms per call, ~20 launches -> ~7
- **in situ: 8.93 -> 9.02 tok/s (+1%)**, far below the +11% micro-bench. The eval-per-call micro-bench
  charges every launch a full sync (~23 us); in the real pipelined decode loop those launches were
  already overlapping, so removing them bought little. Treat micro-bench deltas as upper bounds.
- decode only (m==1): each query has its own key set; prefill falls back. Patch `sa_sdpa` (in DEFAULT_STACK).
- bf16 SDPA variant `sa_sdpa_bf16`: 0.323 ms but 4.9e-4 vs the port (opt-in).

**oMLX confirms this is the production strategy.** Its `wsdpa_attention.py` (411 lines, custom Metal,
fp32 online softmax, sink in the denominator) is *prefill only* — "Only valid for prefill shapes
(L > 1); callers keep decode paths" — and dispatches `grid=(16*128, q_len)`, one head per simdgroup
across 16 threadgroups per query. Occupancy comes from q_len. At decode (q_len=1) oMLX uses stock
SDPA. This is exactly why all three hand-written decode kernels here lost: 64 heads = 64 threadgroups
cannot fill the GPU from inside one launch. Custom attention kernels earn their keep at PREFILL.

**Next lead if prefill ever matters** (it does not for decode-bound agent work): port oMLX's wsdpa
structure (one head per simdgroup, 16 tg/query, online softmax over only the visible window+pool rows)
to V4.1's window+topk layout. Their HISA kernel (two-stage block top-k) is for the indexer, which is 2%.

## DeepSeek-V4.1-Flash: remaining optimisations + native MTP (DSpark) speculative decoding (2026-09-11)

Goal: "explore all remaining conceivable optimisations" after the 9.02 tok/s kernel pass.

### Decode (greedy, output-identical unless noted)

| step | decode tok/s | note |
|---|---|---|
| default stack (previous pass) | 8.99 | |
| + `async` — `mx.async_eval` pipelining of the decode loop | 9.11 | graph build overlaps GPU |
| + `wo_a_f32` — cast the unquantized block-diagonal `wo_a` to fp32 ONCE | 9.66 | the port did `.astype(f32)` on a 134 MB tensor per layer per token (5.4 GB resident) |
| + `gate_f32` — same for the 40 MoE gate weights | **10.10** | |
| + `freqs_pre` — precompute RoPE tables | 9.92 → 9.92 | no effect; kept as harmless |
| + `shared_fuse` — fuse shared-expert w1/w3 into one quantized GEMV | 9.97 (< 10.10), output DIFFERS | wider qmv picks a different tile/accumulation; dropped |

`DEFAULT_STACK` now ends in `nosync,async`; `DEFAULT_POST = "wo_a_f32,gate_f32"` (post-load, needs the
weights). `scripts/turbo/run.py` applies both. Prefill measured at 311 tok/s (920-token prompt).

### Speculative decoding

Verify chunks are nearly free while launch-bound: an n=6 target forward costs 146 ms vs 105 ms for
n=1, n=12 costs 179 ms. So anything that drafts ≥2 correct tokens per step wins.

**Prompt-lookup (n-gram) drafter, no extra model** (`specdec` patch, K=4): 14.31 tok/s on a code-edit
prompt (1.6 tok/step), no gain on prose (1.05 tok/step). Uses the port's `_cache_snapshot/_cache_restore`
for rollback (verified exact).

**Native DSpark/MTP head (the real win).** The release ships 3 draft stages (`mtp.0..2.*`, 14.2 B params,
fp8 attention + fp4 experts + Markov/confidence heads) that the PipeNetwork port drops and no upstream
runtime implements (the mlx-vlm PR stops at weight splitting). The reference `inference/model.py`
specifies the forward (`DSparkAttention`, `DSparkBlock.forward_embed/forward_head`, `forward_spec`);
the verify loop is ours.
- weights: the 3 mtp-only shards of `deepseek-ai/DeepSeek-V4.1-Flash` (`model-00044/45/46-of-00048`,
  8 GB, 27 s download) → `scripts/turbo/mtp_convert.py` → `~/models/DeepSeek-V4.1-Flash-MLX-dspark/`
  (15.4 GB, MLX 8-bit g64; "I8" experts are packed fp4 e2m1 pairs, same as the main model)
- implementation: `scripts/turbo/dspark.py` — each stage: HC block whose attention reads a private 128-slot
  ring of keys made from `main_x = main_norm(main_proj(concat(hc-mean hidden ENTERING layers 37,38,39)))`;
  draft block = `[bonus, noise×4]` (non-causal within the block, SDPA with sink key); stage 2 →
  norm → target head + Markov bias of the previous drafted token. `spec_generate`: draft 5, forward
  `[pending + draft]` through the target in one chunk (capturing the target hiddens), accept the longest
  matching prefix + 1 bonus, roll the cache back on rejection and carry accepted tokens as `pending`.
- drafter quality (isolated, prose): per-position accuracy 77/58/41/27/17 %, 3.03 tokens/step;
  `draft()` = 10.7 ms. On code the acceptance is 75 %.
- **result: code prompt 20.96 tok/s vs 8.49 greedy (2.5×, 4.41 tok/step); prose 11.14 vs 9.59 (+16 %,
  2.46 tok/step).** Memory: 445 GB resident (target 427 + wo_a 5.4 + drafter 15.4).
- confidence head is loaded but unused (greedy verification doesn't need it).

### Numerics finding that matters for any verifier

The port's forward is NOT chunking-invariant in bf16: the same tokens fed 1-at-a-time, in 4-token
chunks, or as one prefill give logits that differ by a median ~1–2.4 (max 10), flipping ~10 % of
low-margin argmaxes. It is not cache rounding (fp32 cache reproduces bf16 exactly) and not my kernels
(plain port shows it too); snapshot/restore is exact. The port's parity tests run in fp32 on a tiny
random config, so they cannot see it. Consequence: a speculative decoder here is "greedy under the
chunk path's numerics", not bit-identical to 1-token greedy — same class of deviation every bf16
spec-decode implementation has, but larger than usual. Perplexity arbitration and an fp32-activation
re-run are recorded below.

**Arbitration (measured).** Perplexity of the same 576 scored tokens: 1-token decode 9.13, 8-token
chunks 9.31, 64-token chunks 9.28, one chunk 9.25, full prefill 9.22 — equivalent. With every bf16
parameter cast to fp32 (fp32 activations) the chunks-vs-single gap drops from median 2.18 to 0.30 and
the argmax flips vanish; the residual is discontinuous routing (MoE top-6, index top-k) flipping on
kernel-order differences, amplified through 40 HC layers in bf16. Each path is deterministic run-to-run.
Verdict: no port bug; the chunk path is a valid verifier.

### Final numbers after the second pass (2026-09-11)

| mode | prose | code-edit |
|---|---|---|
| greedy, default stack + post-load patches | 9.65 tok/s | 8.46 tok/s |
| + native DSpark speculative decoding (`V41_SPEC=1`, default in `scripts/turbo/run.py`) | **12.71** (2.76 tok/step, 35 % accept) | **23.50** (4.55 tok/step, 78 % accept) |

Second-pass changes that made DSpark faster: `sa_sdpa` generalised to m>1 (one query per SDPA batch
row; falls back to the port's chain past m·k > 4096 gathered keys — measured crossover), and
`max_pending=12` (carrying accepted tokens into the next verify chunk costs ~7 ms/token vs ~105 ms
for a separate resync forward).

Where the verify cost goes: n=1 forward 105 ms, n=6 146 ms, n=12 179 ms. The n>1 premium is the
small-M quantized matmul path (`5120x5120` 8-bit: 213 µs at M=1, 280 µs at M=6, 340 µs at M=16 — MLX
already batches small M; nothing cheap left there). The drafter itself is 10.7 ms/step.

Files: `scripts/turbo/dspark.py` (drafter + `spec_generate`), `scripts/turbo/mtp_convert.py`, `scripts/turbo/patches.py`
(`DEFAULT_STACK`, `DEFAULT_POST`, `apply_post_load`), `scripts/turbo/run.py`, diagnostics `scripts/turbo/chunkcheck{2..6}.py`,
`scripts/turbo/dspark_diag.py`, `scripts/turbo/bench{2,3}.py`. Everything under /tmp — copy out before a reboot.
Session end state: oMLX restarted with DeepSeek-V4-Flash (155.8 GB) as the daily driver.

## DeepSeek-V4.1-Flash as a daily driver: server, `pi-deepseekv41`, prefill work (2026-09-11)

### Prefill
Profile of a 512-token chunk (2048-token prompt): **MoE 62 %** (routed experts 57 % — every chunk
re-reads a layer's full 6.8 GB of 4-bit experts, so small chunks are weight-bandwidth-bound),
attention 31 % (the port's gather-based `sparse_attn` 18 %), engram 1 %.
- Chunk size (16k-token prompt): 512 → 285 tok/s, 2048 → 372, 4096 → 382, 8192 → 408. Peak memory
  437 / 442 / 453 GB; the server uses 2048 to stay clear of the 461.9 GB Metal ceiling with the
  15 GB drafter loaded.
- `sparse_attn4` (`v41_metal.py`): one 1024-thread threadgroup = 32 heads of one query; key rows
  staged in threadgroup memory in tiles of 8 and shared by all 32 heads (v3 re-read every row once
  per head — why it lost); online softmax in registers; sink = initial state. vs the port at prefill
  shapes: m=512,k=640: 13.8 → 11.7 ms (short context), 37.1 → 12.8 ms (n=16k), m=4096: 153 → 96 ms.
  Diff vs port ≤ 4e-3 (bf16 ulp). **In situ it LOSES: 12.2k-token prompt through the server, 2048-chunks,
  port 365 tok/s vs v4 330 tok/s (2 runs each).** The micro-bench used random key indices, which punish
  the port's gather (cache-hostile) — real window/top-k lists are localized, and the port's einsum is a
  batched GEMM that a hand-rolled fp32 dot loop with a simd_sum per key cannot match on compute.
  Kept opt-in (`V41_SA_PREFILL=v4`); default is the port path. Lesson: benchmark attention kernels with
  REAL index patterns, and at prefill the answer is simdgroup-matrix GEMM, not per-key dot loops.
  The pi system prompt (19.4k tokens) prefills at ~300-365 tok/s.
- Not done: the MoE prefill path. It needs larger effective batches per expert (bigger chunks are the
  only lever from Python) or a custom gather-GEMM; MLX's `gather_qmm` is what it is.

### Server: `deepseek_v41_mlx/turbo/server.py`
OpenAI-compatible (`/v1/chat/completions` stream+non-stream, `/v1/models`, `/health`), port 8001,
single-threaded on purpose (MLX streams are bound to the thread that first used them: handler
threads died with `There is no Stream(gpu, 4) in current thread`). Greedy + DSpark. Prompt
rendering and tool-call parsing use DeepSeek's own `encoding/encoding.py` from the V4.1 repo
(copied as `v41_encoding.py`): DSML `<｜DSML｜ calls>` tool blocks, `<think>` reasoning, numeric
reasoning effort. `reasoning_effort` in the request → thinking mode (`V41_THINKING` overrides).
- prefix cache: 4 slots keyed by covered tokens; longest common prefix wins; truncation to an even
  position is free (the compressor's open-group carry is derived from `start_pos % ratio`); only
  the tail is prefilled. Measured in a pi session: turn 1 19,409 tokens / 64 s, turn 2 reused
  19,588 of 20,074 (2 s), turn 3 reused 20,262 of 20,336 (0.5 s). Multi-slot exists because pi's
  98-token title request evicted the 20k cache in the single-slot version.
- measured in pi turns: 28–31 tok/s decode with 95–97 % draft acceptance (agent output is very
  predictable), 5.4–5.9 tokens/step.
- smoke tests: `deepseek_v41_mlx/turbo/curl_test.sh` (non-stream, stream, tools, thinking), `v41_slot_test.sh`.

### `~/qwen/bin/pi-deepseekv41`
`pi-deepseekv41 [prompt|pi flags]` ensures the server is up (stops oMLX first — 445 + 156 GB do
not fit), then `exec pi-kimi --lead v41/DeepSeek-V4.1-Flash --subagents v41/DeepSeek-V4.1-Flash`,
so the lead and the fan-out agents run on V4.1 locally while architect/oracle/reviewer/visual stay
on K3 (synthetic). Verbs: `start | stop | status | logs`; `stop` brings oMLX back. pi provider `v41`
was added to `~/.pi/agent/models.json` (contextWindow 131072, maxTokens 16384, reasoning).
First real task (seeded off-by-one in a median function): found, fixed, verified, reported in 4
turns; exit 0.
