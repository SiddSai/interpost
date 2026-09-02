"""Dataset helpers shared by the toxicity-example scripts.

Kept here so probe-fitting and pair-building apply *identical* filtering — the
held-out split for pair building (``skip=`` the probe's training count) only holds
if both stream and filter the same way.
"""

from __future__ import annotations

from datasets import load_dataset


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
