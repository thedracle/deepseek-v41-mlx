# DeepSeek-V4.1-Flash — upstream notes

What `docs/reference/model.py` (DeepSeek's official inference code, the only
implementation of `model_type: deepseek_v41`) actually does, what changed from
V4-Flash, and what surprised us. The MLX port in `deepseek_v41_mlx/` follows
this file, not the HF config's suggestions.

## Headline shape

40 layers, hidden 5120, 64 heads over a single shared 512-d MLA KV vector,
384 routed experts (top-6, sqrt-softplus, route_scale 1.5) + 1 shared expert,
`rms_norm_eps` **1e-20** (effectively zero — matched exactly, not "fixed").
`hub_index` metadata: 510.3 GB. The text stack is **754.6B parameters**, not
~600B: 543.6B routed experts + **196.6B engram tables** + ~14.4B everything
else. The two engram tables are 40% of the checkpoint.

## Attention: sliding window + shared compressed positions

Every layer attends over a 128-token sliding window (ring buffer). Layers with
`compress_ratio > 0` additionally attend over `index_topk` = 512 compressed
positions, selected by an indexer. One `sparse_attn` call takes the
concatenated `[window KV, compressed KV]` with per-query index lists (−1 =
masked); K and V are the same tensor.

Carried over from V4, still load-bearing (negative controls confirm both):

* the **rope inverse** is applied to the attention output tail before the
  grouped output projection;
* per-head **attention sinks** enter the softmax **denominator only** (no
  value vector); an all-masked row yields exactly zero output (finite −1e30
  max floor in the kernel).

Gone from V4: the weightless per-head q RMS-normalization, the compressor
`ape` slot embedding, ratio-4 overlapping spans, and the indexer's Hadamard
rotation. Also gone: **hash routing** (`tid2eid`) — every MoE layer scores.

`wo_a` is still the grouped block-diagonal output LoRA, applied via a reshape
to `[o_groups, o_lora_rank, −1]` and an einsum — so `wo_a` still must never be
MLX-quantized (the reshape would carve packed bytes). The release stores it
fp8-quantized; DeepSeek's own `convert.py` dequantizes it to bf16 for exactly
this reason. Ours does the same.

## Cross-layer sharing (the KV-cache-compression headline)

`compress_ratios` = `[0,0, 2×18, 1×20]` (+ `[0,0,0]` for the three MTP
layers — the 43-entry array covers backbone + MTP).

Only the four `kv_source_layer_ids` [2, 8, 14, 20] own a compressor and a
compressed-KV cache; only the eight `index_source_layer_ids`
[2, 8, 14, 20, 24, 28, 32, 36] own an indexer. The reference threads a
process-global `SharedAttentionRuntime` down the stack; every consumer reads
the most recent producer above it:

| layers | ratio | compressed KV from | top-k selection from | index keys from |
|---|---|---|---|---|
| 0–1 | 0 (window only) | — | — | — |
| 2–7 | 2 | layer 2 | layer 2 | layer 2 |
| 8–13 | 2 | layer 8 | layer 8 | layer 8 |
| 14–19 | 2 | layer 14 | layer 14 | layer 14 |
| 20–23 | 1 | layer 20 | layer 20 | layer 20 |
| 24–27 | 1 | layer 20 | **layer 24** | layer 20 |
| 28–31 | 1 | layer 20 | **layer 28** | layer 20 |
| 32–35 | 1 | layer 20 | **layer 32** | layer 20 |
| 36–39 | 1 | layer 20 | **layer 36** | layer 20 |

So 24/28/32/36 are index sources that own **no keys**: they score with their
own `wq_b`/`weights_proj` against layer 20's key cache (that is why the hub
has 8 × `indexer.wq_b` but only 4 × `indexer.wk`). Layer 20 is additionally
the **candidate source**: it block-pools its scores (blocks of 8, top 2048
blocks, the block holding the query's newest position pinned in) and later
index sources mask their scores with that boolean block mask before their own
top-512 — a two-level top-k. Ratio 1 means "one latent per token": a plain
projection, no gate, no pooling state; ratio 2 pools pairs with a learned
softmax gate in fp32.

Ordering inside a layer matters: the indexer consumes the compressor's latent
**pre-RoPE, pre-quantization** (its keys are derived from that form), then
attention RoPE-rotates the latent at the group's first-token position and
FP4-fake-quantizes it before writing the shared cache.

### Reference decode artifact (bug in model.py)

`Indexer.forward` reads keys through the global pointer `shared_attn.index_k`,
which is only (re)assigned when a layer *publishes* keys. A ratio-2 owner
publishes every second decode step; on the other steps the pointer still holds
the **last** owner's cache — layer 20's — so layers 2/8/14 score their queries
against layer 20's ratio-1 keys on half of all decode steps. Prefill is
unaffected (every owner publishes when `seqlen ≥ ratio`). Measured on the tiny
config: patched-vs-unpatched decode logits differ by 0.67 (relative max). The
MLX port always reads the owner's own cache; the parity test patches the
reference the same way for decode comparison, and separately measures the
artifact.

## QAT activation fake-quantization (all three differ from V4)

The caches hold *fake-quantized* values — this is what the model was trained
to see, and each op was verified bit-exact between the torch stub and MLX:

| what | format | block | scale |
|---|---|---|---|
| window KV (whole 512-d vector, rope tail **included**) | fp8 e4m3 | 32 | ue8m0 (2^ceil(log2(amax/448)), amax ≥ 1e-4) |
| compressed KV latents | fp4 e2m1 | **16** | **e4m3** (RNE of amax/6, amax ≥ 6·2⁻⁹) |
| indexer q and k | fp4 e2m1 | 32 | ue8m0 (amax ≥ 6·2⁻¹²⁶) |

The ue8m0 ceil is computed by IEEE-754 bit manipulation
(`fast_round_scale`), reproduced bit-exactly. FP4 rounding is RNE on
{0, .5, 1, 1.5, 2, 3, 4, 6}: ties at .25/1.25/2.5/5 round down, at
.75/1.75/3.5 round up. A practical consequence for cross-implementation
parity: ~1e-7 upstream noise occasionally lands exactly on a rounding
midpoint and flips one code (verified on a pre-quant pair differing by
1.15e-7 across the −0.033203125 midpoint), and one flipped window entry
cascades down the sequence tail through overlapping windows. All discrete
decisions (expert top-k, index sets, candidate blocks) still agree; with the
QAT sim disabled on both sides, agreement is ~1e-6 everywhere. A useful
side-effect: every fake-quantized value is exactly representable in bf16
(≤5 mantissa bits × power-of-two-ish scales), so bf16 caches are lossless.

## Hyper-Connections: V4's machinery, staggered

Same Sinkhorn split as V4 (sigmoid pre + eps, 2·sigmoid post, comb row-softmax
then 20 alternating column/row normalizations with eps inside the divisions),
but the coefficients are consumed one sub-layer **late**: attention's collapse
uses the *previous layer's FFN* `pre` (identity one-hot on copy 0 at layer 0),
the FFN's collapse uses this layer's attention `pre`, and the LM head's final
collapse uses the last layer's FFN `pre`. There is no separate `hc_head_fn`
(V4 had one). `hc_post` still computes
`out[k] = post[k]·x + Σ_j comb[j,k]·residual[j]` — residual indexed by the
summed axis.

## Engram (new): 197B parameters of n-gram memory

Layers 1 and 14 add a gated n-gram lookup to the hc-expanded stream before the
block. Per position: the 2-, 3-, 4-gram ending there, each hashed into 8
heads; every (n-gram size, head) owns a disjoint prime-sized bucket range,
primes drawn sequentially upward from `engram_vocab_size` = 16,000,000 and
never reused. **Verified: the sum of each layer's 24 primes equals its
`engram_num_embeddings` entry exactly (384,006,168 and 384,016,682).**

* Hashing runs over a **compressed token map** (NFKC/NFD, strip accents,
  lowercase, whitespace collapse — " The"/"the"/"THE" collide): 129,280 →
  99,092 ids, built from the release tokenizer at first load and cached.
  Multipliers are odd int64s from `default_rng(10007·layer_id)`, bounded so
  `token·multiplier` cannot overflow; the running XOR after i lookbacks is the
  (i+1)-gram hash. Lookback stops at sequence start (and image spans); blocked
  slots read the compressed pad token (id 2's image).
* The 24 fetched rows (256-d, stored fp8 with per-32 ue8m0 row scales,
  dequantized per lookup) are projected by `wkv` into one key per hc copy plus
  one shared value; the gate is `sigmoid(copysign(sqrt(|dot|)), dot)` of a
  per-copy normalized stream·key dot. Decode needs the previous 3 compressed
  ids — the cache carries a per-sequence id history.

## MoE details

Scores = `sqrt(softplus(x/gate_temp))` in fp32. `gate.bias` reorders the top-6
only — weights come from the unbiased scores, normalized by `sum + 1e-20`
(literally 1e-20, not `norm_eps`) and scaled ×1.5. The checkpoint carries
`gate.bias_vl`, used **instead of** `gate.bias` for tokens inside image spans
(training's `noaux_tc_for_vl`); text-only inference always uses `gate.bias`,
but `bias_vl` is kept loaded so checkpoints round-trip. Clamped SwiGLU
(limit 10): up two-sided, gate upper-only, fp32.

## Rope

Compressing layers (ratio > 0): YaRN (factor 16, original 65,536) on
`compress_rope_theta` 160,000. Window-only layers (0, 1): plain theta 10,000,
YaRN off. One table per layer serves queries, window KV, compressed latents
(group j at position j·ratio), the indexer, and the inverse on the way out.
Adjacent-pair (real, imag) convention, last 64 channels only.

## Checkpoint formats (differ from V4)

* fp8 weights: e4m3 `[out, in]` + ue8m0 scale `[⌈out/32⌉, ⌈in/32⌉]`
  (**32×32** blocks; V4 was 128×128), tensor named `<w>.scale` (not
  `weight_scale_inv` — the hub already uses inference-style names throughout:
  no `model.` prefix, `attn`/`ffn`, `gate.bias`).
* fp4 experts: e2m1 two per byte `[out, in/2]` (low nibble = even index; code
  8 = "−0" decodes to +0.0) + ue8m0 `[out, in/32]`.
* engram tables: fp8 rows + ue8m0 `[rows, 8]`.
* `mx.load` maps safetensors F8_E4M3 / F8_E8M0 to uint8, which is exactly
  what the dequant kernels want. All three decode paths verified bit-exact
  against the reference semantics.
* **Confirmed against the completed real download (48/48 shards, 475 GiB):**
  fp8 weights uint8 with `[⌈out/32⌉,⌈in/32⌉]` uint8 scales, experts int8
  `[out, in/2]` with `[out, in/32]` scales, `attn_sink`/`hc_*`/`gate.bias*`
  fp32, `gate.weight` bf16, engram tables `[384,006,168 × 256]` uint8 + 
  `[rows, 8]` scales in two single-tensor ~101 GB shards (47/48 — DeepSeek
  ships >50 GB files), engram `q/k_weight` bf16. Layers 0 and 1 strict-load
  from the raw shards; the release tokenizer compresses to exactly **99,092**
  ids, matching `engram_compressed_vocab_size`.
* MLX gotcha: evaluating any *slice* of a lazily-loaded tensor materializes
  the whole load node — chunking over a 98 GB mmapped engram table bounds
  nothing. The converter therefore row-slices the tables straight from the
  safetensors file (`quantize_engram_table_from_file`); the ladder accepts
  the transient whole-table residency instead (fine on 512 GB, freed after
  the layer).

## Dropped / passthrough

`mtp.{0,1,2}.*` (DSpark speculative stack: 3 draft layers with 128-expert
MoEs, a Markov head over token ids, a confidence head) hangs off the separate
`forward_spec` path — dropped in conversion, counted. The vision tower
(`vision.*`, `aligner.*`, `image_start/end/newline`, all bf16, ~0.4B params)
is carried through conversion **unchanged** for a future VL runtime; `load.py`
reports it as declared passthrough rather than silently ignoring it.

## Chunked prefill

The reference implements only full prefill and 1-token decode (its decode
branches assume `seqlen == 1`). The MLX port's forward is chunk-general —
window ring, compressor partial-group state, indexer visibility, and the
engram id history all carry across arbitrary chunk boundaries — validated
against single-shot prefill and against the reference's own longer prefill.

## Quantization recipe (sizes computed, quality untested until weights land)

Expert share of affine-quantizable parameters: **98.8%**. Must stay
unquantized: `wo_a` (reshape), `hc_*`, `attn_sink`, `gate.weight`/biases,
compressor projections, `indexer.wk`/`weights_proj`, norms. Quantizable:
attention projections, `indexer.wq_b`, shared experts, routed experts,
embed/head, `engram.wkv`, and the engram tables (native fp8 or re-quantized).

| build (g64) | size | fits |
|---|---:|---|
| experts 8b, rest 8b, engram native fp8 | 791 GB | 1 TB only |
| experts 6b, rest 6b, engram native fp8 | 654 GB | 1 TB only |
| experts 4b, rest 8b, engram native fp8 | 519 GB | nothing (512 GB is too small) |
| experts 4b, rest 8b, **engram 6b** | 476 GB | 512 GB |
| experts 4b, rest 8b, **engram 4b** | 427 GB | 512 GB ⭐ |
| experts 4b, rest 4b, engram 4b | 424 GB | 512 GB (saves only 3 GB) |

Mixed-4_8 over uniform 4-bit costs ~3 GB — the V4 lesson (non-expert weights
~50× more quantization-sensitive) almost certainly transfers. The open
empirical question is the engram tables: native fp8+ue8m0 is ~8.25 bits and
optimal at that width (MLX 8-bit affine is *larger*); 4-bit halves 203 GB to
111 GB. They are additive, sigmoid-gated residual contributions, which argues
for tolerance, but measure ppl at engram 4b vs 6b vs native before publishing.
V4's warning stands: experts at 3-bit collapsed outright; assume the same
cliff here.
