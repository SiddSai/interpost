"""Phase 1a: mask-aware pooling."""

import pytest
import torch

from interpost.activations.pooling import POOLING_MODES, pool


@pytest.fixture
def hidden():
    # (B=2, S=3, D=2): row 0 tokens = [10,11,12] pattern, row 1 = [20,21,22]
    return torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )


def test_rejects_bad_mode(hidden):
    with pytest.raises(ValueError):
        pool(hidden, torch.ones(2, 3), mode="median")


def test_rejects_shape_mismatch(hidden):
    with pytest.raises(ValueError):
        pool(hidden, torch.ones(2, 4), mode="mean")
    with pytest.raises(ValueError):
        pool(torch.ones(2, 3), torch.ones(2, 3), mode="mean")  # 2-D hidden


@pytest.mark.parametrize("mode", [m for m in POOLING_MODES if m != "per_token"])
def test_output_shape(hidden, mode):
    out = pool(hidden, torch.ones(2, 3), mode=mode)
    assert out.shape == (2, 2)


def test_per_token_is_identity(hidden):
    out = pool(hidden, torch.zeros(2, 3), mode="per_token")
    assert torch.equal(out, hidden)


def test_mean_respects_mask(hidden):
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    out = pool(hidden, mask, mode="mean")
    torch.testing.assert_close(out, torch.tensor([[1.5, 1.5], [5.5, 5.5]]))


def test_last_picks_last_kept_token(hidden):
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    out = pool(hidden, mask, mode="last")
    torch.testing.assert_close(out, torch.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_max_respects_mask(hidden):
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = pool(hidden, mask, mode="max")
    torch.testing.assert_close(out, torch.tensor([[2.0, 2.0], [6.0, 6.0]]))


@pytest.mark.parametrize("mode", ["mean", "last", "max"])
def test_empty_mask_row_pools_to_zero(hidden, mode):
    mask = torch.tensor([[0, 0, 0], [1, 1, 1]])
    out = pool(hidden, mask, mode=mode)
    torch.testing.assert_close(out[0], torch.zeros(2))


def test_gradients_flow_through_mean(hidden):
    h = hidden.clone().requires_grad_(True)
    pool(h, torch.ones(2, 3), mode="mean").sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
