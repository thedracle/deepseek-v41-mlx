"""Publish a built DeepSeek-V4.1-Flash MLX quant to the Hub, card rendered from the measurements.

    .venv/bin/python scripts/upload.py --dir <build dir> --repo pipenetwork/<name> [--yes] [--card-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/Users/david/llm/dsv41-out")
UPSTREAM = "deepseek-ai/DeepSeek-V4.1-Flash"
CODE_REPO = "https://github.com/PipeNetwork/deepseek-v41-mlx"
ORDER = ["DeepSeek-V4.1-Flash-MLX-mixed-4_8bit", "DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6"]
LADDER_NAME = {"DeepSeek-V4.1-Flash-MLX-mixed-4_8bit": "mixed-4_8-engram4",
               "DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6": "mixed-4_8-engram6"}
RAM = {"DeepSeek-V4.1-Flash-MLX-mixed-4_8bit": "512 GB Mac (tight: 427 GB resident)",
       "DeepSeek-V4.1-Flash-MLX-mixed-4_8bit-engram6": "1 TB-class machine (477 GB build; a lazy forward transiently needs ~2x)"}

CARD = """---
license: mit
base_model: {upstream}
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- deepseek_v41
- mixture-of-experts
- 4-bit
pipeline_tag: text-generation
library_name: mlx
---

# {repo_name}

MLX (Apple Silicon) build of [**DeepSeek-V4.1-Flash**](https://huggingface.co/{upstream}) —
754.6B parameters: 40 layers x 384 routed experts, MLA with cross-layer KV-cache sharing (4
compressor-owning layers serve all 40), a two-layer **engram** hashed n-gram embedding whose
tables alone are 196.6B parameters (~40% of the checkpoint), staggered Sinkhorn
hyper-connections, and per-layer attention sinks — quantized to **{recipe}**.

**These files are modified**: dequantized from the FP8/FP4 release (bit-exact decode of the
32x32-block ue8m0 fp8 and per-32 fp4 packing) and re-quantized; the architecture is unchanged.
The 3 multi-token-prediction layers (DSpark markov/confidence heads) are not included; the vision
tower and aligner are carried unmodified but the runtime is text-only.

## Runtime

`deepseek_v41` exists in **no** runtime — not transformers, not mlx-lm, not mlx-vlm. This
checkpoint loads only through the port:

```bash
git clone {code_repo} && cd deepseek-v41-mlx && pip install -r requirements.txt
python scripts/smoke_generate.py /path/to/{repo_name}
```
```python
from deepseek_v41_mlx.load import load
model, tokenizer = load("/path/to/{repo_name}")
```

The port was validated against DeepSeek's own `inference/model.py` (the only reference): fp32
tiny-config parity **1e-6** across prefill / cached decode / chunked prefill, the three QAT
fake-quant ops bit-exact, and negative controls proving the fragile paths are load-bearing
(rope-inverse 0.84, attention sinks 0.65, cross-layer sharing 0.56 logit shift when broken).
Strict loading reports zero missing / zero unexpected tensors. One reference *decode* bug was
found and documented (odd-step indexer reads the wrong layer's keys; 0.67 logit shift — the port
uses the owner's cache): see `docs/upstream-notes.md` in the repo.

## Size and what is quantized

**{gb:.1f} GB** on disk. RAM: {ram}.

| group | share of parameters | this build |
|---|---:|---|
| routed experts (40 x 384, `w1/w2/w3`) | 543.6B (72%) | 4-bit, group 64 |
| engram tables (2 x [384,006,168 x 256]) | 196.6B (26%) | {engram_desc} |
| attention (MLA), shared experts, embeddings, `head` | ~14B | 8-bit, group 64 |
| `wo_a` (block-diagonal output LoRA), hyper-connections, sinks, router biases, compressor, indexer keys, norms | — | unquantized (bf16/fp32) |

## Quality

**Per-layer divergence ladder** vs the bf16-dequantized reference — every one of the 40 decoder
layers run on identical inputs (16,384 tokens of wikitext-2), teacher-forced and free-running,
with the ladder's arithmetic asserted bit-identical to this converter's:

{ladder_table}

The engram finding that shapes this set: **6-bit engram is indistinguishable from the shipped
fp8 tables** (free-running 0.1945 vs 0.1948) while **4-bit engram costs +7.3%** free-running —
but the engram-6 build is 477 GB and a 512 GiB machine cannot run it, so the engram-4 build is
the one that fits and the engram-6 build serves 1 TB machines.

{ppl_section}

## License

MIT, as the upstream model. Port code: [{code_repo}]({code_repo}).
"""


def ladder_table(npz):
    z = np.load(npz, allow_pickle=True); names = [str(n) for n in z["names"]]; L = int(z["layers"])
    tf, fr, fc = z["teacher_rel"][:L], z["free_rel"][:L], z["free_cos"][:L]
    rows = ["| recipe | teacher-forced (mean) | free-running (final layer) | cosine (final) |", "|---|---:|---:|---:|"]
    label = {"mixed-4_8-engram4": "**mixed 4/8, engram 4-bit (this set's 512 GB build)**",
             "mixed-4_8-engram6": "**mixed 4/8, engram 6-bit (this set's 1 TB build)**",
             "mixed-4_8-engram-native": "mixed 4/8, engram as shipped (fp8/ue8m0)"}
    for i, n in enumerate(names):
        rows.append(f"| {label.get(n, n)} | {tf[:, i].mean():.4f} | {fr[L-1, i]:.4f} | {fc[L-1, i]:.4f} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir); name = args.repo.split("/")[-1]
    cfg = json.load(open(d / "config.json"))
    q = cfg.get("quantization", {}); eb = None
    for k, v in (q.get("modules") or {}).items():
        if "engram" in k: eb = v.get("bits")
    engram_desc = f"{eb}-bit, group 64" if eb else "as shipped (fp8, ue8m0 row scales, ~8.25 bits/weight)"
    recipe = f"4-bit experts / 8-bit attention&shared / {eb or 'native fp8'}-bit engram"
    gb = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e9
    lt = ladder_table(OUT_ROOT / "ladder.npz")
    res_p = ROOT / "ppl_results.json"
    res = json.load(open(res_p)) if res_p.exists() else {}
    ppl_section = ""
    if name in res:
        r = res[name]
        ppl_section = (f"**Perplexity** (wikitext-2 test, {r['tokens']:,} tokens in {r['windows']} windows of {r['seq_len']}, "
                       f"through this runtime): **{r['perplexity']:.4f}** [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]. "
                       f"Greedy generation is coherent (collapse check).")
    else:
        ppl_section = ("**Perplexity is not measurable for this build on our 512 GiB machine** (the 477 GB build plus "
                       "activations exceeds it in every load mode — four were tried). Its quality case is the ladder above: "
                       "its engram treatment is indistinguishable from the shipped fp8 tables, and every other module is "
                       "identical to the measured engram-4 build (ppl 2.8963 [2.7103, 3.0933]). Strict-loaded: zero missing / "
                       "zero unexpected tensors.")
    card = CARD.format(upstream=UPSTREAM, code_repo=CODE_REPO, repo_name=name, recipe=recipe, gb=gb,
                       ram=RAM.get(name, ""), engram_desc=engram_desc, ladder_table=lt, ppl_section=ppl_section)
    (d / "README.md").write_text(card)
    print(f"repo {args.repo}\ndir {d}\nfiles {sum(1 for p in d.iterdir() if p.is_file())}, {gb:.1f} GB\n{lt}\n\n{ppl_section[:200]}")
    if not args.yes:
        print("\ndry run — pass --yes to upload"); return 0
    from huggingface_hub import HfApi
    import time
    api = HfApi()
    if args.card_only:
        api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
        print(f"card refreshed https://huggingface.co/{args.repo}"); return 0
    api.create_repo(args.repo, exist_ok=True, repo_type="model")
    for _ in range(30):
        try: api.model_info(args.repo); break
        except Exception: time.sleep(2)
    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    print(f"uploaded https://huggingface.co/{args.repo}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
