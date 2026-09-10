"""Minimal greedy generation for DeepSeek-V4.1 MLX."""

from __future__ import annotations

import mlx.core as mx

from .model import Model


def greedy_generate(model: Model, input_ids, max_new_tokens: int = 64,
                    max_seq_len: int | None = None, eos_id: int = 1,
                    dtype=mx.float32):
    """input_ids: list[int] or [1, n] array. Returns the generated ids."""
    ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
    total = ids.shape[1] + max_new_tokens
    cache = model.make_cache(bsz=ids.shape[0],
                             max_seq_len=max_seq_len or (total + 8), dtype=dtype)
    logits = model(ids, cache, last_logit_only=True)
    out = []
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(max_new_tokens):
        t = int(tok[0])
        if t == eos_id:
            break
        out.append(t)
        logits = model(tok[:, None], cache, last_logit_only=True)
        tok = mx.argmax(logits[:, -1], axis=-1)
    return out


def load_tokenizer(path: str):
    import os
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
