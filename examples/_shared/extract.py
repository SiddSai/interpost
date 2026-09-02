"""Batched activation extraction and candidate generation for probe training.

Teacher-forced: we run one forward pass over ``prompt + response`` and pool the
residual stream over the *response* token span. This matches "mean activation across
generated tokens (excluding the prompt)" from Wehner & Fritz closely enough for a
reproduction, without per-step generation-time hooks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from interpost.activations import HookManager, pool


def _encode_pairs(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad ``prompt + response`` sequences. Returns (input_ids, attention_mask,
    response_mask), each ``(B, S)``; ``response_mask`` is 1 only on response tokens."""
    rows: list[tuple[list[int], list[int]]] = []
    for prompt, response in zip(prompts, responses, strict=True):
        p_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        r_ids = tokenizer(response, add_special_tokens=False).input_ids
        r_ids = r_ids[: max(0, max_length - len(p_ids))]
        rows.append((p_ids, r_ids))

    width = max(len(p) + len(r) for p, r in rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    response_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (p_ids, r_ids) in enumerate(rows):
        seq = p_ids + r_ids
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
        response_mask[i, len(p_ids) : len(seq)] = 1
    return input_ids, attention_mask, response_mask


@torch.no_grad()
def extract_pooled(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    responses: Sequence[str],
    layers: Sequence[int],
    *,
    pooling: str = "mean",
    batch_size: int = 16,
    max_length: int = 512,
    device: str | torch.device | None = None,
) -> dict[int, np.ndarray]:
    """Return ``{layer: (N, D) float32 array}`` of response-pooled activations."""
    model.eval()
    device = device or next(model.parameters()).device
    hooks = HookManager(model)
    layers = list(layers)
    chunks: dict[int, list[np.ndarray]] = {li: [] for li in layers}

    for start in range(0, len(prompts), batch_size):
        p = prompts[start : start + batch_size]
        r = responses[start : start + batch_size]
        input_ids, attn, resp_mask = _encode_pairs(tokenizer, p, r, max_length)
        input_ids, attn, resp_mask = input_ids.to(device), attn.to(device), resp_mask.to(device)
        with hooks.capture(layers) as acts:
            model(input_ids=input_ids, attention_mask=attn)
        for li in layers:
            pooled = pool(acts[li].float(), resp_mask, pooling)  # (B, D)
            chunks[li].append(pooled.detach().cpu().numpy())

    return {li: np.concatenate(v, axis=0) for li, v in chunks.items()}


@torch.no_grad()
def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    k: int = 5,
    temperature: float = 1.0,
    do_sample: bool = True,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    seed: int = 0,
) -> list[list[str]]:
    """Generate ``k`` continuations per prompt. Returns ``list[list[str]]`` (only the
    newly generated text, prompt stripped). ``do_sample=False`` forces greedy (k is
    then effectively 1)."""
    model.eval()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    out: list[list[str]] = []
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_id, num_return_sequences=k)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        gen = model.generate(**enc, **gen_kwargs)
        prompt_len = enc["input_ids"].shape[1]
        new_tokens = gen[:, prompt_len:].reshape(len(batch), k, -1)
        for i in range(len(batch)):
            out.append(
                [tokenizer.decode(new_tokens[i, j], skip_special_tokens=True) for j in range(k)]
            )
    return out
