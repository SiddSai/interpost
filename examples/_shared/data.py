"""Dataset helpers shared by the toxicity-example scripts.

Kept here so probe-fitting and pair-building apply *identical* filtering — the
held-out split for pair building (``skip=`` the probe's training count) only holds
if both stream and filter the same way.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset

PKU_REPO = "PKU-Alignment/PKU-SafeRLHF"


def stream_civil_comments_prompts(
    *,
    min_toxicity: float = 0.5,
    min_chars: int = 40,
    max_chars: int = 300,
    skip: int = 0,
    limit: int,
) -> list[str]:
    """Return ``limit`` Civil Comments texts (own toxicity >= ``min_toxicity``,
    length-filtered), after skipping the first ``skip`` that qualify."""
    raw = load_dataset("google/civil_comments", split="train", streaming=True)
    out: list[str] = []
    seen = 0
    for row in raw:
        text = row["text"].strip().replace("\n", " ")
        if not (min_chars <= len(text) <= max_chars):
            continue
        if row["toxicity"] < min_toxicity:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _pku_rows(split: str, limit: int | None):
    ds = load_dataset(PKU_REPO, split=split)
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def load_pku_probe_examples(
    split: str = "train", *, limit: int | None = None
) -> tuple[list[str], list[str], np.ndarray]:
    """Flatten PKU-SafeRLHF into (prompts, responses, labels) for probe fitting.

    label = 1 if the response is unsafe (harmful compliance), 0 if safe (refusal /
    benign). One row yields up to two examples (response_0, response_1).
    """
    prompts: list[str] = []
    responses: list[str] = []
    labels: list[int] = []
    for row in _pku_rows(split, limit):
        for i in (0, 1):
            prompts.append(row["prompt"])
            responses.append(row[f"response_{i}"])
            labels.append(0 if row[f"is_response_{i}_safe"] else 1)
    return prompts, responses, np.asarray(labels, dtype=int)


def load_pku_preference_pairs(
    split: str = "train",
    *,
    limit: int | None = None,
    seed: int = 0,
    conversational: bool = True,
) -> Dataset:
    """PKU-SafeRLHF preference pairs where exactly one response is safe — chosen =
    the safe one, rejected = the unsafe one.

    ``conversational=True`` (for instruct models) emits messages lists so TRL applies
    the chat template; otherwise plain ``prompt`` / ``chosen`` / ``rejected`` strings.
    """
    rows = {"prompt": [], "chosen": [], "rejected": []}
    for row in _pku_rows(split, limit):
        s0, s1 = row["is_response_0_safe"], row["is_response_1_safe"]
        if s0 == s1:
            continue  # both safe or both unsafe -> no clean safety signal
        safe_i = 0 if s0 else 1
        prompt, chosen = row["prompt"], row[f"response_{safe_i}"]
        rejected = row[f"response_{1 - safe_i}"]
        if conversational:
            prompt = [{"role": "user", "content": prompt}]
            chosen = [{"role": "assistant", "content": chosen}]
            rejected = [{"role": "assistant", "content": rejected}]
        rows["prompt"].append(prompt)
        rows["chosen"].append(chosen)
        rows["rejected"].append(rejected)
    return Dataset.from_dict(rows).shuffle(seed=seed)
