"""Per-layer damage for every build in the ladder, against the dequantized-native
bf16 reference, in one streaming pass over the raw V4.1 release.

Same idea as glm53-mlx's ladder, adapted to V4.1's stateful architecture: a
layer is not a pure function of its hidden input — the hyper-connection stream
carries a staggered ``pre_mix``, and attention layers share compressed KV /
index selections / candidate masks down the stack. So:

* **teacher-forced** — the variant layer sees the *reference's* upstream world:
  the bf16 hidden state, the bf16 ``pre_mix``, and the bf16 lane's shared
  caches (its compressed KV, index keys, top-k, candidates), snapshotted before
  the reference layer runs. The variant's own cache writes go to a scratch
  layer cache so the reference lane is never polluted. This isolates one
  layer's weight damage.
* **free-running** — every variant carries its OWN StreamState-equivalent
  (caches, shared pointers, pre_mix, hidden): a quantized layer 2 produces
  quantized latents/keys/selections that layers 3..7 then consume. This is
  what actually happens at inference and is where compounding shows.

Variant arithmetic is exactly the converter's: expert/other widths through
``load.quant_predicate`` (the same predicate ``load.py`` replays), engram
treatment row-locally identical to ``convert.quantize_engram_table`` (affine
quantization is row-local, so quantizing only the gathered rows is exact) —
both asserted at startup against the convert-path outputs on real layer-0 /
layer-1 tensors.

    .venv/bin/python scripts/eval_ladder.py --src ../DeepSeek-V4.1-Flash-src \\
        --out ladder.npz --variants 8bit,6bit,mixed-4_8-engram-native,mixed-4_8-engram6,mixed-4_8-engram4,4bit-engram4

Resumable at layer boundaries: results go to --out after every layer, lane
state (hidden, pre_mix, shared selections, source caches) to <out>.state.npz;
--resume continues from there.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepseek_v41_mlx.cache import LayerCache, ModelCache
from deepseek_v41_mlx.convert import quantize_engram_table, quantize_tree, sanitize_group
from deepseek_v41_mlx.dequant import dequant_fp8_rows
from deepseek_v41_mlx.engram import EngramHasher
from deepseek_v41_mlx.hyper_connections import make_identity_pre_mix
from deepseek_v41_mlx.load import load_token_map, quant_predicate
from deepseek_v41_mlx.model import Block, SharedState
from deepseek_v41_mlx.stream import load_subset, read_config, shard_map

try:
    mx.set_wired_limit(int(470e9))
except Exception as e:  # noqa: BLE001
    print("[warn]", e, flush=True)


def parse_variant(spec: str) -> tuple[str, int, int, str]:
    """'8bit' -> (8,8,native); 'mixed-4_8-engram4' -> (4,8,'4');
    '4bit-engram4' -> (4,4,'4'); 'mixed-4_8-engram-native' -> (4,8,'native')."""
    name = spec.strip()
    rest, engram = name, "native"
    m = re.search(r"-engram-?(native|\d+)$", rest)
    if m:
        engram, rest = m.group(1), rest[:m.start()]
    if rest.startswith("mixed-"):
        body = rest[len("mixed-"):]
        body = body[:-3] if body.endswith("bit") else body
        e, o = body.split("_")
        ebits, obits = int(e), int(o)
    else:
        ebits = obits = int(rest[:-3] if rest.endswith("bit") else rest)
    return name, ebits, obits, engram


class RowQuantEngram(nn.Module):
    """Native fp8 table, gathered rows re-quantized through MLX affine —
    row-local, so numerically identical to convert.quantize_engram_table."""

    def __init__(self, weight_u8: mx.array, scale_u8: mx.array, bits: int, group_size: int):
        super().__init__()
        self.weight, self.scale = weight_u8, scale_u8
        self.bits, self.group_size = bits, group_size

    def __call__(self, idx: mx.array) -> mx.array:
        rows = dequant_fp8_rows(self.weight[idx], self.scale[idx])
        shape = rows.shape
        flat = rows.astype(mx.bfloat16).reshape(-1, shape[-1])
        w, s, b = mx.quantize(flat, group_size=self.group_size, bits=self.bits)
        out = mx.dequantize(w, scales=s, biases=b,
                            group_size=self.group_size, bits=self.bits)
        return out.reshape(shape).astype(mx.float32)


class Lane:
    """One propagation lane: hidden stream + caches + shared pointers + pre_mix."""

    def __init__(self, margs, bsz: int, seq: int):
        self.cache = ModelCache(margs, bsz, max_seq_len=seq, dtype=mx.bfloat16)
        self.shared = SharedState()
        self.pre_mix = make_identity_pre_mix(bsz, seq, margs.hc_mult)
        self.h: mx.array | None = None


class _CacheView:
    """A ModelCache whose entry at one layer is scratch — everything else
    aliases the reference lane's caches (read-only for consumers)."""

    def __init__(self, ref_cache: ModelCache, li: int, scratch: LayerCache):
        self.layers = list(ref_cache.layers)
        self.layers[li] = scratch


def run_lane(layer: Block, li: int, lane: Lane, hashes, margs) -> mx.array:
    h = lane.h.astype(mx.bfloat16)
    if layer.engram is not None and hashes is not None:
        h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
    out, lane.pre_mix = layer(h, lane.pre_mix, 0, lane.cache, lane.shared)
    return out.astype(mx.float32)


def run_teacher(layer: Block, li: int, h_in, pre_in, shared_snap, ref_cache,
                margs, hashes, bsz: int, seq: int) -> mx.array:
    scratch = LayerCache(bsz, margs, li, seq, dtype=mx.bfloat16)
    sh = SharedState()
    sh.kv_src_cache, sh.index_src_cache, sh.topk_idxs, sh.candidates = shared_snap
    h = h_in.astype(mx.bfloat16)
    if layer.engram is not None and hashes is not None:
        h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
    out, _ = layer(h, pre_in, 0, _CacheView(ref_cache, li, scratch), sh)
    return out.astype(mx.float32)


def divergence(got: mx.array, want: mx.array) -> tuple[float, float]:
    """Per-token relative L2 and cosine over the flattened hc*dim vector."""
    a = got.reshape(-1, got.shape[-2] * got.shape[-1]).astype(mx.float32)
    b = want.reshape(-1, want.shape[-2] * want.shape[-1]).astype(mx.float32)
    diff = mx.sqrt(mx.sum((a - b) ** 2, axis=-1))
    norm = mx.sqrt(mx.sum(b * b, axis=-1))
    cos = mx.sum(a * b, axis=-1) / mx.maximum(
        mx.sqrt(mx.sum(a * a, axis=-1)) * norm, 1e-20)
    rel = mx.mean(diff / mx.maximum(norm, 1e-20))
    mx.eval(rel, cos)
    return float(rel.item()), float(mx.mean(cos).item())


def build_block(margs, li: int, sane: dict, prefix: str) -> Block:
    layer = Block(li, margs)
    layer.load_weights([(k[len(prefix):], v) for k, v in sane.items()], strict=True)
    return layer


def make_variant(margs, li: int, sane: dict, prefix: str, ebits: int, obits: int,
                 engram: str, group_size: int) -> Block:
    layer = build_block(margs, li, sane, prefix)
    if layer.engram is not None and engram != "native":
        e = layer.engram.embed
        layer.engram.embed = RowQuantEngram(e.weight, e.scale, int(engram), group_size)
    nn.quantize(layer, group_size=group_size, bits=obits,
                class_predicate=quant_predicate(group_size, obits, ebits))
    return layer


# ---- resumable lane state ------------------------------------------------

def _shared_ids(margs, upto: int):
    """(kv_src, index_owner) layer ids live after layers [0, upto) have run."""
    kv = [s for s in margs.kv_source_layers if s < upto]
    owners = [s for s in margs.kv_source_layers
              if s < upto and s in margs.index_source_layers]
    return (max(kv) if kv else -1), (max(owners) if owners else -1)


def dump_lanes(path, lanes: dict, margs, upto: int):
    blob = {}
    for key, lane in lanes.items():
        blob[f"{key}|h"] = np.asarray(lane.h.astype(mx.float32))
        blob[f"{key}|pre"] = np.asarray(lane.pre_mix.astype(mx.float32))
        if lane.shared.topk_idxs is not None:
            blob[f"{key}|topk"] = np.asarray(lane.shared.topk_idxs)
        if lane.shared.candidates is not None:
            blob[f"{key}|cand"] = np.asarray(lane.shared.candidates).astype(np.uint8)
        for s in margs.kv_source_layers:
            if s < upto:
                lc = lane.cache.layers[s]
                blob[f"{key}|comp{s}"] = np.asarray(lc.comp_kv.astype(mx.float32))
                if lc.index_k is not None:
                    blob[f"{key}|idxk{s}"] = np.asarray(lc.index_k.astype(mx.float32))
    np.savez(path, **blob)


def restore_lanes(path, keys, margs, bsz: int, seq: int, upto: int) -> dict:
    blob = np.load(path)
    lanes = {}
    kv_id, owner_id = _shared_ids(margs, upto)
    for key in keys:
        lane = Lane(margs, bsz, seq)
        lane.h = mx.array(blob[f"{key}|h"])
        lane.pre_mix = mx.array(blob[f"{key}|pre"])
        if f"{key}|topk" in blob:
            lane.shared.topk_idxs = mx.array(blob[f"{key}|topk"])
        if f"{key}|cand" in blob:
            lane.shared.candidates = mx.array(blob[f"{key}|cand"].astype(bool))
        for s in margs.kv_source_layers:
            if f"{key}|comp{s}" in blob:
                lc = lane.cache.layers[s]
                lc.comp_kv = mx.array(blob[f"{key}|comp{s}"]).astype(mx.bfloat16)
                if f"{key}|idxk{s}" in blob:
                    lc.index_k = mx.array(blob[f"{key}|idxk{s}"]).astype(mx.bfloat16)
        if kv_id >= 0:
            lane.shared.kv_src_cache = lane.cache.layers[kv_id]
        if owner_id >= 0:
            lane.shared.index_src_cache = lane.cache.layers[owner_id]
        lanes[key] = lane
    return lanes


# ---- startup self-check: ladder arithmetic == converter arithmetic -------

def self_check(margs, sane: dict, prefix: str, li: int, variants, group_size: int):
    """Every variant's ladder arithmetic must equal the converter's, module
    for module; the engram row check runs on the first engram layer for each
    engram-quantizing variant (affine quantization is row-local, so gathered
    rows are exactly the converted table's rows)."""
    checked = []
    for name, ebits, obits, engram in variants:
        layer = make_variant(margs, li, sane, prefix, ebits, obits, engram, group_size)
        for mod, tname in ((layer.attn.wq_a, f"{prefix}attn.wq_a.weight"),
                           (layer.ffn.experts.gate_proj,
                            f"{prefix}ffn.experts.gate_proj.weight")):
            qt, _ = quantize_tree({tname: sane[tname]}, obits, ebits, group_size)
            base = tname[:-len(".weight")]
            for suffix, got in (("weight", mod.weight), ("scales", mod.scales),
                                ("biases", mod.biases)):
                want = qt[f"{base}.{suffix}"]
                assert mx.array_equal(got, want).item(), f"{name}: {base}.{suffix} differs"
        if layer.engram is not None and engram != "native":
            idx = mx.arange(min(256, layer.engram.embed.weight.shape[0]))
            got = layer.engram.embed(idx)
            t = quantize_engram_table(layer.engram.embed.weight[idx],
                                      layer.engram.embed.scale[idx],
                                      int(engram), group_size)
            want = mx.dequantize(t["weight"], scales=t["scales"], biases=t["biases"],
                                 group_size=group_size, bits=int(engram)).astype(mx.float32)
            assert mx.array_equal(got, want).item(), \
                f"{name}: engram row-quant differs from converter"
            checked.append(f"{name}:engram")
        checked.append(name)
        del layer
        mx.clear_cache()
    print(f"  self-check @layer {li}: ladder quantization == converter arithmetic "
          f"({', '.join(checked)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "DeepSeek-V4.1-Flash-src"))
    ap.add_argument("--out", default="ladder.npz")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=0, help="samples per state group (0 = all at once)")
    ap.add_argument("--mode", choices=("teacher", "free", "both"), default="both")
    ap.add_argument("--ids", default=str(Path(__file__).resolve().parents[1] / "ppl_corpus.npy"))
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--no-engram", action="store_true",
                    help="strip engram from every lane incl. the reference "
                         "(for runs before the engram shards finish downloading)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    src, out_path = Path(args.src), Path(args.out)
    state_path = Path(str(out_path) + ".state.npz")
    variants = [parse_variant(v) for v in args.variants.split(",")]
    names = [v[0] for v in variants]
    raw_cfg, margs = read_config(src)
    if args.no_engram:
        margs = dataclasses.replace(margs, engram_layer_ids=(), engram_num_embeddings=())
    try:
        smap = shard_map(src)
    except FileNotFoundError:
        # the release index downloads last; the vendored copy is identical
        import json as _json
        ref_idx = Path(__file__).resolve().parents[1] / "docs" / "reference" / "hub_index.json"
        smap = _json.load(open(ref_idx))["weight_map"]
        print(f"[warn] {src}/model.safetensors.index.json missing; using vendored hub_index.json",
              flush=True)
    if args.no_engram:
        # never touch the engram shards (they download last and may be absent)
        smap = {k: v for k, v in smap.items() if ".engram." not in k}

    ids_np = np.load(args.ids)[: args.samples * args.seq_len]
    ids_np = ids_np.astype(np.int64).reshape(args.samples, args.seq_len)
    B = args.batch or args.samples
    chunks = [(i, min(i + B, args.samples)) for i in range(0, args.samples, B)]
    n_layers = args.limit_layers or margs.n_layers
    seq = args.seq_len
    free_on = args.mode in ("free", "both")
    teacher_on = args.mode in ("teacher", "both")

    # engram hashes (identical for every lane; None when engram out of scope)
    hashes_by_chunk = [None] * len(chunks)
    if margs.engram_layer_ids and any(li < n_layers for li in margs.engram_layer_ids):
        token_map = load_token_map(str(src), margs,
                                   cache_dir=str(Path(__file__).resolve().parents[1]))
        hasher = EngramHasher(margs, token_map)
        for ci, (a, b) in enumerate(chunks):
            scratch = np.zeros((b - a, seq), dtype=np.int64)
            hashes_by_chunk[ci] = mx.array(hasher(ids_np[a:b], 0, scratch))

    lane_keys = [f"ref|{ci}" for ci in range(len(chunks))]
    if free_on:
        lane_keys += [f"{n}|{ci}" for n in names for ci in range(len(chunks))]

    tf_rel = np.zeros((n_layers, len(variants)))
    tf_cos = np.ones((n_layers, len(variants)))
    fr_rel = np.zeros((n_layers, len(variants)))
    fr_cos = np.ones((n_layers, len(variants)))
    start_layer = 0

    if args.resume and out_path.exists() and state_path.exists():
        prev = np.load(out_path, allow_pickle=False)
        assert list(prev["names"]) == names and int(prev["seq_len"]) == seq \
            and int(prev["samples"]) == args.samples, "resume config mismatch"
        done = int(prev["layers"])
        tf_rel[:done] = prev["teacher_rel"][:done]
        tf_cos[:done] = prev["teacher_cos"][:done]
        fr_rel[:done] = prev["free_rel"][:done]
        fr_cos[:done] = prev["free_cos"][:done]
        start_layer = done
        lanes = restore_lanes(state_path, lane_keys, margs, B, seq, done)
        print(f"resuming at layer {done}", flush=True)
    else:
        embed = load_subset(src, smap, "embed.weight")["embed.weight"]
        lanes = {}
        for key in lane_keys:
            ci = int(key.rsplit("|", 1)[1])
            a, b = chunks[ci]
            lane = Lane(margs, b - a, seq)
            h0 = embed[mx.array(ids_np[a:b])].astype(mx.float32)
            lane.h = mx.broadcast_to(h0[:, :, None, :],
                                     (b - a, seq, margs.hc_mult, h0.shape[-1]))
            mx.eval(lane.h)
            lanes[key] = lane
        del embed
        mx.clear_cache()

    print(f"tokens/chunk: {[(b - a) * seq for a, b in chunks]}  variants: {names}  "
          f"layers {start_layer}..{n_layers - 1}  engram "
          f"{'OFF' if args.no_engram else 'on'}", flush=True)

    started = time.time()
    for li in range(start_layer, n_layers):
        t0 = time.time()
        prefix = f"layers.{li}."
        raw = load_subset(src, smap, prefix)
        sane = sanitize_group(sorted(raw), raw)
        for k, v in sane.items():
            if ".engram.embed." not in k:   # the tables stay mmapped, never whole
                mx.eval(v)

        if not args.no_self_check and (
                li == start_layer or
                (li in margs.engram_layer_ids and
                 any(v[3] != "native" for v in variants))):
            self_check(margs, sane, prefix, li, variants, args.group_size)

        ref_layer = build_block(margs, li, sane, prefix)
        ref_out, snaps = {}, {}
        for ci in range(len(chunks)):
            lane = lanes[f"ref|{ci}"]
            snaps[ci] = (lane.h, lane.pre_mix,
                         (lane.shared.kv_src_cache, lane.shared.index_src_cache,
                          lane.shared.topk_idxs, lane.shared.candidates))
            ref_out[ci] = run_lane(ref_layer, li, lane, hashes_by_chunk[ci], margs)
            mx.eval(ref_out[ci])
            lane.h = ref_out[ci]
        del ref_layer
        mx.clear_cache()

        for vi, (name, ebits, obits, engram) in enumerate(variants):
            layer = make_variant(margs, li, sane, prefix, ebits, obits, engram,
                                 args.group_size)
            from mlx.utils import tree_flatten
            mx.eval([v for p, v in tree_flatten(layer.parameters())
                     if "engram.embed" not in p])

            if teacher_on:
                acc = []
                for ci, (a, b) in enumerate(chunks):
                    h_in, pre_in, shared_snap = snaps[ci]
                    got = run_teacher(layer, li, h_in, pre_in, shared_snap,
                                      lanes[f"ref|{ci}"].cache, margs,
                                      hashes_by_chunk[ci], b - a, seq)
                    acc.append((divergence(got, ref_out[ci]), (b - a) * seq))
                    del got
                w = sum(t for _, t in acc)
                tf_rel[li, vi] = sum(d[0] * t for d, t in acc) / w
                tf_cos[li, vi] = sum(d[1] * t for d, t in acc) / w

            if free_on:
                acc = []
                for ci, (a, b) in enumerate(chunks):
                    lane = lanes[f"{name}|{ci}"]
                    got = run_lane(layer, li, lane, hashes_by_chunk[ci], margs)
                    mx.eval(got)
                    acc.append((divergence(got, ref_out[ci]), (b - a) * seq))
                    lane.h = got
                w = sum(t for _, t in acc)
                fr_rel[li, vi] = sum(d[0] * t for d, t in acc) / w
                fr_cos[li, vi] = sum(d[1] * t for d, t in acc) / w

            del layer
            mx.clear_cache()

        del sane, raw, ref_out, snaps
        mx.clear_cache()

        np.savez(out_path, names=np.array(names), teacher_rel=tf_rel,
                 teacher_cos=tf_cos, free_rel=fr_rel, free_cos=fr_cos,
                 layers=np.array(li + 1), tokens=np.array(ids_np.size),
                 seq_len=np.array(seq), samples=np.array(args.samples))
        dump_lanes(state_path, lanes, margs, li + 1)

        rate = (time.time() - started) / (li - start_layer + 1)
        report = "  ".join(f"{n}:{tf_rel[li, i]:.4f}/{fr_rel[li, i]:.4f}"
                           for i, n in enumerate(names))
        print(f"layer {li:2d}/{n_layers} {time.time() - t0:6.1f}s  "
              f"eta {rate * (n_layers - li - 1) / 60:5.1f}m  tf/free  {report}",
              flush=True)

    print(f"\n{'variant':>26} {'teacher rel (mean)':>19} {'free rel (final)':>17} {'free cos':>10}")
    for i, name in enumerate(names):
        print(f"{name:>26} {tf_rel[start_layer:n_layers, i].mean():>19.5f} "
              f"{fr_rel[n_layers - 1, i]:>17.5f} {fr_cos[n_layers - 1, i]:>10.5f}")
    print(f"\nwrote {out_path}  ({(time.time() - started) / 60:.1f} min, "
          f"peak {mx.get_peak_memory() / 1e9:.0f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
