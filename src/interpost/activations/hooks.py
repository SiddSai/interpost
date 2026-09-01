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

# Paths to the decoder-layer ``nn.ModuleList``, tried in order, across architectures.
_LAYER_PATHS: tuple[str, ...] = (
    "model.layers",  # Llama, Qwen2/3, Mistral, Gemma
    "transformer.h",  # GPT-2, GPT-J, Falcon
    "gpt_neox.layers",  # Pythia / GPT-NeoX
)


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
        """Return the decoder-layer ``ModuleList``, unwrapping PEFT / ``base_model``."""
        roots = [model]
        for attr in ("base_model", "model"):
            inner = getattr(model, attr, None)
            if isinstance(inner, nn.Module):
                roots.append(inner)

        for root in roots:
            for path in _LAYER_PATHS:
                obj: object = root
                for part in path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                    return obj

        raise AttributeError(
            f"could not locate decoder layers on {type(model).__name__}; "
            f"add its path to interpost.activations.hooks._LAYER_PATHS"
        )

    @contextmanager
    def capture(self, layers: list[int]) -> Iterator[dict[int, torch.Tensor]]:
        """Yield ``{layer_index: (B, S, D) hidden states}``, populated after the
        forward pass runs inside the ``with`` block. Captured tensors retain grad.
        """
        raise NotImplementedError("HookManager.capture is implemented in Phase 1")

    @contextmanager
    def inject(
        self, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Iterator[None]:
        """Replace layer ``layer``'s output hidden states with ``fn(hidden)`` for
        the duration of the ``with`` block (used for steering-style interventions).
        """
        raise NotImplementedError("HookManager.inject is implemented in Phase 3")
