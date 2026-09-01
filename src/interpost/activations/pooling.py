"""Mask-aware pooling of per-token hidden states to one vector per sequence.

Signature is fixed in Phase 0 so callers can be written against it; the body lands
in Phase 1.
"""

from __future__ import annotations

import torch

POOLING_MODES: tuple[str, ...] = ("last", "mean", "max", "per_token")


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool ``hidden`` over the sequence axis using ``mask``.

    Args:
        hidden: ``(B, S, D)`` hidden states.
        mask: ``(B, S)`` 1 for real tokens, 0 for padding / prompt tokens to skip.
        mode: one of :data:`POOLING_MODES`. ``"per_token"`` returns the masked
            ``(B, S, D)`` tensor unchanged (callers pool later).

    Returns:
        ``(B, D)`` for ``last`` / ``mean`` / ``max``; ``(B, S, D)`` for ``per_token``.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"mode must be one of {POOLING_MODES}, got {mode!r}")
    raise NotImplementedError("pool() is implemented in Phase 1")
