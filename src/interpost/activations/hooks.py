"""HookManager: register forward hooks on a causal LM's decoder layers to capture
or edit residual-stream activations.

Phase 0: layer resolution is implemented (it is just plumbing and is testable);
``capture`` lands in Phase 1 and ``inject`` in Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

# Attributes that wrap the "real" model (PEFT, accelerate, HF); walked breadth-first.
_WRAPPER_ATTRS: tuple[str, ...] = ("base_model", "model", "module", "transformer", "gpt_neox")
# ModuleList attribute names that hold the decoder blocks.
_LAYERS_ATTRS: tuple[str, ...] = ("layers", "h")


class HookManager:
    """Attach and detach forward hooks on a Hugging Face causal LM's decoder layers.

    Args:
        model: a ``transformers`` causal-LM (or a PEFT-wrapped one).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers: nn.ModuleList = self._resolve_layers(model)

    @staticmethod
    def _resolve_layers(model: nn.Module) -> nn.ModuleList:
        """BFS through wrapper attrs (PEFT / accelerate / HF nesting) for the first
        non-empty decoder-block ``ModuleList``."""
        seen: set[int] = set()
        queue: list[nn.Module] = [model]
        while queue:
            obj = queue.pop(0)
            if id(obj) in seen or not isinstance(obj, nn.Module):
                continue
            seen.add(id(obj))
            for attr in _LAYERS_ATTRS:
                cand = getattr(obj, attr, None)
                if isinstance(cand, nn.ModuleList) and len(cand) > 0:
                    return cand
            for attr in _WRAPPER_ATTRS:
                queue.append(getattr(obj, attr, None))
        raise AttributeError(
            f"could not locate decoder layers on {type(model).__name__}; "
            f"extend interpost.activations.hooks._WRAPPER_ATTRS / _LAYERS_ATTRS"
        )

    @contextmanager
    def capture(
        self, layers: list[int], *, first: bool = False
    ) -> Iterator[dict[int, torch.Tensor]]:
        """Yield ``{layer_index: (B, S, D) hidden states}``, populated once the
        forward pass runs inside the ``with`` block.

        The dict is keyed by the exact indices passed in (negative indices are
        allowed and kept as-is). Captured tensors are the live layer outputs, so
        they stay attached to the autograd graph — do not use with gradient
        checkpointing, where the hook also fires during backward recomputation.

        ``first=True`` keeps only the first forward per layer and ignores later
        ones — use it when the ``with`` block triggers several forwards (e.g. a
        DPO step's policy forward followed by a no-grad reference forward) and you
        want the first (grad-carrying, policy) activations.
        """
        if not layers:
            raise ValueError("capture() needs at least one layer index")
        n = len(self.layers)
        for li in layers:
            if not -n <= li < n:
                raise IndexError(f"layer {li} out of range for a {n}-layer model")

        store: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(key: int):
            def _hook(_module, _inputs, output):
                if first and key in store:
                    return
                store[key] = output[0] if isinstance(output, tuple) else output

            return _hook

        try:
            for li in layers:
                handles.append(self.layers[li].register_forward_hook(_make_hook(li)))
            yield store
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def inject(
        self, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Iterator[None]:
        """Replace layer ``layer``'s output hidden states with ``fn(hidden)`` for
        the duration of the ``with`` block (used for steering-style interventions).
        """
        raise NotImplementedError("HookManager.inject is implemented in Phase 3")
