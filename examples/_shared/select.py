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

# (prompts, responses) -> badness array. Both lists are flat and parallel: one entry
# per candidate, with the prompt repeated across its k candidates. A response-only
# scorer (e.g. an output classifier) just ignores the prompts argument.
Scorer = Callable[[list[str], list[str]], np.ndarray]


def select_pairs(
    prompts: Sequence[str],
    candidates: Sequence[Sequence[str]],
    scorer: Scorer,
    *,
    min_gap: float = 0.0,
) -> Dataset:
    """Build a preference ``Dataset`` (columns: ``prompt``, ``chosen``, ``rejected``,
    ``chosen_score``, ``rejected_score``).

    Args:
        prompts: one prompt per row.
        candidates: ``k`` candidate responses per prompt (``k >= 2``).
        scorer: ``(flat_prompts, flat_responses) -> 1-D badness array``.
        min_gap: skip a prompt if ``max(score) - min(score) <= min_gap`` (no usable
            preference signal).
    """
    prompts = list(prompts)
    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    spans: list[tuple[int, int]] = []
    for prompt, cand in zip(prompts, candidates, strict=True):
        cand = list(cand)
        if len(cand) < 2:
            raise ValueError("need at least 2 candidates per prompt")
        spans.append((len(flat_responses), len(flat_responses) + len(cand)))
        flat_prompts.extend([prompt] * len(cand))
        flat_responses.extend(cand)

    scores = np.asarray(scorer(flat_prompts, flat_responses), dtype=np.float64)
    if scores.shape != (len(flat_responses),):
        raise ValueError(f"scorer returned {scores.shape}, expected {(len(flat_responses),)}")

    rows: dict[str, list] = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "chosen_score": [],
        "rejected_score": [],
    }
    for prompt, (lo, hi) in zip(prompts, spans, strict=True):
        s = scores[lo:hi]
        if float(s.max() - s.min()) <= min_gap:
            continue
        best, worst = lo + int(s.argmin()), lo + int(s.argmax())
        rows["prompt"].append(prompt)
        rows["chosen"].append(flat_responses[best])
        rows["rejected"].append(flat_responses[worst])
        rows["chosen_score"].append(float(scores[best]))
        rows["rejected_score"].append(float(scores[worst]))

    return Dataset.from_dict(rows)
