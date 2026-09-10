"""REAP-prune an already-built (quantized) V4.1 MLX checkpoint, streaming,
at several ratios at once.

Pruning the quantized build is equivalent to pruning bf16 and requantizing:
expert subsetting runs along the expert axis (0) of the stacked
``experts.{gate,up,down}_proj.{weight,scales,biases}`` tensors while affine
quant groups run along the input dim — independent axes. Router rows
(``gate.weight``, ``gate.bias``, ``gate.bias_vl``) subset the same way.
Never-routed experts prune first. Shards containing no expert/router tensors
(top level, the engram table shards, vision) are byte-copied.

The quantization module map in config.json is keyed by module *paths*, which
subsetting does not change, so it carries over intact; ``n_routed_experts``
is corrected inside text_config (the nested value wins in ModelArgs).

    .venv/bin/python scripts/prune_build.py <SRC_BUILD> <saliency.npz> <ratio,ratio,...>
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import mlx.core as mx

SRC = sys.argv[1].rstrip("/")
SAL = sys.argv[2]
RATIOS = [int(r) for r in (sys.argv[3] if len(sys.argv) > 3 else "25,37,50").split(",")]
_LAYER_RE = re.compile(r"^layers\.(\d+)\.ffn\.")
_PER_EXPERT = ("experts.gate_proj.", "experts.up_proj.", "experts.down_proj.")
_ROUTER = ("gate.weight", "gate.bias", "gate.bias_vl")
_SHARD_CAP = 10_000_000_000


def keep_indices(usage, ratio):
    sal = usage["saliency"].astype(np.float64)
    counts = usage["counts"].astype(np.float64)
    E = sal.shape[1]
    K = int(round(E * (1 - ratio / 100)))
    S = np.where(counts == 0, -1.0, sal)      # never-routed experts prune first
    order = np.argsort(-S, axis=1)
    keep = np.sort(order[:, :K], axis=1)
    retained = np.array([np.take_along_axis(sal[i], keep[i], 0).sum() /
                         max(sal[i].sum(), 1e-9) for i in range(sal.shape[0])])
    return keep, K, retained


def needs_subset(name):
    m = _LAYER_RE.match(name)
    if m is None:
        return None
    rest = name[m.end():]
    if rest.startswith(_PER_EXPERT) or rest in _ROUTER:
        return int(m.group(1))
    return None


def main():
    usage = np.load(SAL)
    E = usage["saliency"].shape[1]
    outputs = {}
    for r in RATIOS:
        keep, K, retained = keep_indices(usage, r)
        base = os.path.basename(SRC)
        assert "-MLX-" in base, base
        dst = os.path.join(os.path.dirname(SRC), base.replace("-MLX-", f"-REAP{r}-MLX-"))
        outputs[r] = dict(keep=keep, K=K, dst=dst, idx={}, buf={}, bytes=0, sid=0,
                          retained=retained)
        print(f"[prune] REAP{r}: keep {K}/{E} in {usage['saliency'].shape[0]} MoE layers | "
              f"saliency retained mean {100 * retained.mean():.1f}% "
              f"worst {100 * retained.min():.1f}% -> {dst}", flush=True)
        os.makedirs(dst, exist_ok=True)

    def flush(st):
        if not st["buf"]:
            return
        st["sid"] += 1
        fn = f"model-{st['sid']:05d}.safetensors"
        mx.save_safetensors(os.path.join(st["dst"], fn), st["buf"],
                            metadata={"format": "mlx"})
        for k in st["buf"]:
            st["idx"][k] = fn
        st["buf"], st["bytes"] = {}, 0

    index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    by_shard = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)

    for shard in sorted(by_shard):
        names = by_shard[shard]
        if not any(needs_subset(n) is not None for n in names):
            # engram tables / top level / vision: byte-copy, no rewrite
            for r, st in outputs.items():
                dst_f = os.path.join(st["dst"], shard)
                if not os.path.exists(dst_f):
                    try:
                        os.link(os.path.join(SRC, shard), dst_f)   # same volume: instant
                    except OSError:
                        shutil.copy2(os.path.join(SRC, shard), dst_f)
                for n in names:
                    st["idx"][n] = shard
            print(f"[prune] {shard} (linked)", flush=True)
            continue
        tensors = mx.load(os.path.join(SRC, shard))
        for name, arr in tensors.items():
            li = needs_subset(name)
            for r, st in outputs.items():
                out = arr[mx.array(st["keep"][li])] if li is not None else arr
                mx.eval(out)
                st["buf"][name] = out
                st["bytes"] += out.nbytes
                if st["bytes"] >= _SHARD_CAP:
                    flush(st)
        del tensors
        mx.clear_cache()
        print(f"[prune] {shard}", flush=True)

    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    for r, st in outputs.items():
        flush(st)
        total = sum(os.path.getsize(os.path.join(st["dst"], f)) for f in set(st["idx"].values()))
        json.dump({"metadata": {"total_size": total}, "weight_map": st["idx"]},
                  open(os.path.join(st["dst"], "model.safetensors.index.json"), "w"), indent=2)
        cfg = json.loads(json.dumps(src_cfg))
        if "text_config" in cfg:
            cfg["text_config"]["n_routed_experts"] = st["K"]
        cfg["n_routed_experts"] = st["K"]
        cfg["reap"] = {"kept_experts": st["K"], "original_experts": int(E), "ratio_pct": r,
                       "moe_layers": int(usage["saliency"].shape[0]),
                       "saliency_retained_mean": float(st["retained"].mean()),
                       "saliency_retained_worst": float(st["retained"].min()),
                       "calibration_tokens": int(usage["tokens"]),
                       "saliency": os.path.basename(SAL)}
        json.dump(cfg, open(os.path.join(st["dst"], "config.json"), "w"), indent=2)
        for f in os.listdir(SRC):
            if (f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json")
                    and ".manifest." not in f) or f == "LICENSE":
                shutil.copy2(os.path.join(SRC, f), os.path.join(st["dst"], f))
        print(f"[prune] wrote {st['dst']} ({st['sid']} rewritten shards + copies, "
              f"{total / 1e9:.1f} GB)", flush=True)
    print("[prune] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
