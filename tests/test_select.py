"""Phase 1b/1c: signal-selected preference pairs."""

import numpy as np
import pytest

from examples._shared.select import select_pairs


def _length_badness(prompts, responses):
    return np.array([len(r) for r in responses], dtype=float)


def test_chosen_is_lowest_score_rejected_is_highest():
    prompts = ["p1", "p2"]
    candidates = [
        ["short", "a much longer response", "mid one"],
        ["tiny", "medium length", "the longest response here by far"],
    ]
    ds = select_pairs(prompts, candidates, _length_badness)
    assert ds["chosen"] == ["short", "tiny"]
    assert ds["rejected"] == ["a much longer response", "the longest response here by far"]
    assert ds["prompt"] == ["p1", "p2"]


def test_scorer_receives_prompt_repeated_per_candidate():
    seen = {}

    def scorer(prompts, responses):
        seen["prompts"] = list(prompts)
        seen["responses"] = list(responses)
        return np.arange(len(responses), dtype=float)

    select_pairs(["A", "B"], [["a1", "a2"], ["b1", "b2", "b3"]], scorer)
    assert seen["prompts"] == ["A", "A", "B", "B", "B"]
    assert seen["responses"] == ["a1", "a2", "b1", "b2", "b3"]


def test_min_gap_skips_flat_prompts():
    prompts = ["p1", "p2"]
    candidates = [["aa", "bb", "cc"], ["x", "yyyy"]]  # p1 all length 2 -> skipped
    ds = select_pairs(prompts, candidates, _length_badness, min_gap=0.0)
    assert ds["prompt"] == ["p2"]


def test_scores_are_recorded():
    ds = select_pairs(["p"], [["a", "abcde"]], _length_badness)
    assert ds["chosen_score"] == [1.0]
    assert ds["rejected_score"] == [5.0]


def test_rejects_too_few_candidates():
    with pytest.raises(ValueError):
        select_pairs(["p"], [["only one"]], _length_badness)


def test_rejects_bad_scorer_shape():
    with pytest.raises(ValueError):
        select_pairs(["p"], [["a", "b"]], lambda prompts, responses: np.zeros(1))
