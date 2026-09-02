"""Signal-selected preference pairs — the Wehner & Fritz "probe-based DPO" recipe,
and interpost's Mode-1 `relabel`/`select` primitive in embryo.

Given k candidate responses per prompt and a scorer that returns *badness* (higher =
worse: more toxic / more harmful), take the lowest-scoring candidate as ``chosen`` and
the highest as ``rejected``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from datasets import Dataset

Scorer = Callable[[list[str]], np.ndarray]


def select_pairs(
    prompts: Sequence[str],
    candidates: Sequence[Sequence[str]],
    scorer: Scorer,
    *,
    min_gap: float = 0.0,
) -> Dataset:
    """Build a preference ``Dataset`` (columns: ``prompt``, ``chosen``, ``rejected``).

    Args:
        prompts: one prompt per row.
        candidates: ``k`` candidate responses per prompt (``k >= 2``).
        scorer: maps a flat list of response strings to a 1-D badness array.
        min_gap: skip a prompt if ``max(score) - min(score) <= min_gap`` (no usable
            preference signal).
    """
    prompts = list(prompts)
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for cand in candidates:
        cand = list(cand)
        if len(cand) < 2:
            raise ValueError("need at least 2 candidates per prompt")
        spans.append((len(flat), len(flat) + len(cand)))
        flat.extend(cand)

    scores = np.asarray(scorer(flat), dtype=np.float64)
    if scores.shape != (len(flat),):
        raise ValueError(f"scorer returned {scores.shape}, expected {(len(flat),)}")

    rows = {"prompt": [], "chosen": [], "rejected": [], "chosen_score": [], "rejected_score": []}
    for prompt, (lo, hi) in zip(prompts, spans, strict=True):
        s = scores[lo:hi]
        if float(s.max() - s.min()) <= min_gap:
            continue
        best, worst = lo + int(s.argmin()), lo + int(s.argmax())
        rows["prompt"].append(prompt)
        rows["chosen"].append(flat[best])
        rows["rejected"].append(flat[worst])
        rows["chosen_score"].append(float(scores[best]))
        rows["rejected_score"].append(float(scores[worst]))

    return Dataset.from_dict(rows)
