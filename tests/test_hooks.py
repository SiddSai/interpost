"""Phase 1a: HookManager.capture, tested against a tiny fake decoder stack
(no model download)."""

import pytest
import torch
from torch import nn

from interpost.activations.hooks import HookManager


class _Block(nn.Module):
    """Adds a per-layer constant and returns a tuple, like an HF decoder layer."""

    def __init__(self, bump: float):
        super().__init__()
        self.bump = bump

    def forward(self, x: torch.Tensor):
        return (x + self.bump,)


class _Inner(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.layers = nn.ModuleList(_Block(float(i + 1)) for i in range(n))


class _FakeLM(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.model = _Inner(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.model.layers:
            x = blk(x)[0]
        return x


def test_resolve_layers_on_fake_lm():
    hm = HookManager(_FakeLM(4))
    assert len(hm.layers) == 4


def test_capture_keys_and_shapes():
    model = _FakeLM(4)
    hm = HookManager(model)
    x = torch.zeros(2, 3, 5)
    with hm.capture([0, 2, -1]) as acts:
        model(x)
    assert set(acts) == {0, 2, -1}
    for v in acts.values():
        assert v.shape == (2, 3, 5)


def test_capture_values_are_the_layer_outputs():
    model = _FakeLM(4)
    hm = HookManager(model)
    x = torch.zeros(1, 1, 1)
    with hm.capture([0, 1, 3]) as acts:
        model(x)
    # cumulative bumps: layer i output = sum(1..i+1)
    assert acts[0].item() == 1.0
    assert acts[1].item() == 1.0 + 2.0
    assert acts[3].item() == 1.0 + 2.0 + 3.0 + 4.0


def test_hooks_are_removed_after_context():
    model = _FakeLM(3)
    hm = HookManager(model)
    with hm.capture([0]):
        model(torch.zeros(1, 1, 1))
    assert all(len(layer._forward_hooks) == 0 for layer in hm.layers)


def test_capture_tensors_retain_grad():
    model = _FakeLM(3)
    hm = HookManager(model)
    x = torch.zeros(1, 2, 4, requires_grad=True)
    with hm.capture([1]) as acts:
        out = model(x)
    out.sum().backward()
    assert acts[1].requires_grad
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_out_of_range_layer_raises():
    hm = HookManager(_FakeLM(4))
    with pytest.raises(IndexError):
        with hm.capture([4]):
            pass
    with pytest.raises(ValueError):
        with hm.capture([]):
            pass
