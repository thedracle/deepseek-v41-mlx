# deepseek-v41-mlx

MLX (Apple Silicon) runtime and quantization tooling for
[**deepseek-ai/DeepSeek-V4.1-Flash**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) —
754.6B parameters (`model_type: deepseek_v41`): 40 layers × 384 routed experts, MLA with
**cross-layer KV-cache sharing** (four compressor-owning layers serve all forty — the release's
"KV cache compression" headline), a two-layer **engram** hashed n-gram embedding whose tables are
196.6B parameters (~40% of the checkpoint, two single ~101.5 GB tensors), staggered Sinkhorn
hyper-connections, per-layer attention sinks, and QAT-simulated cache quantization.

Published builds: **[pipenetwork/DeepSeek-V4.1-Flash MLX](https://huggingface.co/pipenetwork)** —
`mixed-4_8bit` (427 GB, 512 GB Mac, wikitext-2 ppl 2.8963) and `mixed-4_8bit-engram6`
(477 GB, 1 TB machines).

## Why this exists

`deepseek_v41` is carried by **no runtime** — not transformers, not mlx-lm, not mlx-vlm. The only
reference is DeepSeek's own `inference/model.py` (vendored under `docs/reference/`). This package
is a from-scratch port built on our validated
[DeepSeek-V4-Flash port](https://github.com/PipeNetwork/deepseek-v4-mlx), keeping V4's landmines
(the rope **inverse** on attention output, sinks in the softmax **denominator only**, the
block-diagonal `wo_a` that must never be quantized, Sinkhorn hyper-connections indexed by the
summed axis) and adding what V4.1 changed:

* **Cross-layer sharing**: layers 2/8/14/20 own KV compressors and index keys; every other layer
  consumes the most recent owner's caches (layer 20 serves 21–39). Index-source layers 24–36
  score their own queries against **layer 20's** keys, masked by layer 20's candidate blocks.
* **Engram** (layers 1, 14): 2/3/4-grams × 8 heads hashed over a 99,092-id accent/case/whitespace-
  compressed token map (derived from the tokenizer at load), tables shipped fp8 with ue8m0 row
  scales. Decode threads compressed-id history through the cache.
* Hash routing is gone; the router is sqrt-softplus with separate text/image-span biases
  (`bias` / `bias_vl`); `rms_norm_eps` is 1e-20 and is matched, not "fixed".
* New packing: fp8 in 32×32 blocks with ue8m0 scales, experts fp4 two-per-byte — decoded
  bit-exactly (unit-tested round-trips).

**A reference decode bug was found**: `Indexer.forward` reads keys through a pointer reassigned
only when a source layer publishes, so on odd decode steps layers 2/8/14 score against layer 20's
keys — a 0.67 relative logit shift vs consistent semantics. The port uses the owner's own cache;
`docs/upstream-notes.md` has the details, plus the operational Metal-watchdog rules learned at
this scale (a lazy MLX load node materializes **whole** on first eval; after one GPU timeout the
process's further GPU submissions are silently ignored).

## Validation

```bash
./.venv/bin/python tests/test_parity.py
```

fp32 tiny-config parity vs the reference: prefill/decode/chunked-prefill all ≤ 1.1e-6; the three
QAT fake-quant ops bit-exact (including e4m3 midpoints); negative controls — rope-inverse broken
0.84, sinks zeroed 0.65, cross-layer sharing severed 0.56; strict release-layout round-trips at
bf16 and quantized; the streaming layer-at-a-time pass bit-equal to the loaded model. On the real
checkpoint: strict load with zero missing / zero unexpected tensors (vision tower + aligner
passthrough; 3 MTP layers dropped), coherent greedy generation at 427 GB resident.

## Measurements

Per-layer divergence ladder (16,384 tokens, teacher-forced + free-running, the ladder's
arithmetic asserted bit-identical to the converter's; `scripts/eval_ladder.py`, resumable —
resume verified bit-identical across a kv-source boundary):

| recipe | tf mean | free@39 | cos@39 |
|---|---:|---:|---:|
| 8bit (791 GB) | 0.0084 | 0.1243 | 0.9910 |
| 6bit (654 GB) | 0.0177 | 0.1393 | 0.9886 |
| mixed 4/8, engram native | 0.0335 | 0.1948 | 0.9800 |
| **mixed 4/8, engram 6-bit** (477 GB) | 0.0335 | **0.1945** | 0.9801 |
| **mixed 4/8, engram 4-bit** (427 GB) | 0.0342 | 0.2090 | 0.9775 |
| 4bit, engram 4-bit (424 GB) | 0.0579 | 0.2714 | 0.9634 |

The engram tables decide the set: **6-bit engram is free** (indistinguishable from the shipped
fp8), **4-bit engram costs +7.3% free-running** — but the engram-6 build cannot run on a 512 GiB
machine (measured: four load modes, all fail — a lazy forward transiently needs ~2× build size,
and macOS compresses MLX's dirty buffers rather than evicting cache). So: engram-4 is the 512 GB
build (ppl **2.8963** [2.7103, 3.0933] over 286,580 tokens, coherent smoke), engram-6 is the
1 TB build, and 8/6-bit exist only as ladder rows. Uniform 4-bit is dominated (3 GB smaller,
30% worse free-running).

**REAP-pruned tiers** (`scripts/reap_calibrate.py` on the resident quantized build, `scripts/prune_build.py`
pruning the quantized tensors directly — expert axis vs quant-group axis, exactly equivalent to
prune-then-requantize; engram shards hardlinked across builds): REAP25 (288/384 experts, 351 GB) costs
**×1.0281** [1.018, 1.039] paired vs the unpruned build — the V4 family prunes nearly free at 25% — while
REAP50 (275 GB) costs ×1.1681 with visibly degraded generation style; REAP37 (314 GB) was built and
measured (×1.0766) but not published — it serves no RAM tier REAP25 doesn't. Split-half calibration
agreement 88–93%.

## Layout

| path | what |
|---|---|
| `deepseek_v41_mlx/` | the runtime: model, cross-layer cache plumbing, engram, dequant (fp8/ue8m0 + fp4), strict loader, streaming |
| `scripts/convert_cli.py` | per-group streaming converter (38 GB peak for a 754B model), `--resume`, row-streamed engram re-quantization |
| `scripts/eval_ladder.py` | per-layer divergence ladder with per-variant StreamState lanes |
| `scripts/ppl_large.py`, `ppl_compare.py`, `ppl_corpus.py`, `calib_corpus.py` | perplexity harness (paired bootstrap) |
| `scripts/smoke_generate.py`, `check_strict_load.py` | chunked-prefill greedy generation; strict-load gate |
| `scripts/upload.py`, `make_collection.py` | publishing, cards rendered from measurements |
| `tests/test_parity.py` | the validation above |
| `docs/reference/` | DeepSeek's inference code (the reference) |
| `docs/upstream-notes.md` | the decode bug + Metal/MLX operational findings |
