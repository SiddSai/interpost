"""Mask-aware pooling of per-token hidden states to one vector per sequence."""

from __future__ import annotations

import torch

POOLING_MODES: tuple[str, ...] = ("last", "mean", "max", "per_token")


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool ``hidden`` over the sequence axis using ``mask``.

    Args:
        hidden: ``(B, S, D)`` hidden states.
        mask: ``(B, S)`` with 1 for tokens to include, 0 for padding / prompt
            tokens to skip. Any dtype; treated as boolean-ish.
        mode: one of :data:`POOLING_MODES`.

    Returns:
        ``(B, D)`` for ``"last"`` / ``"mean"`` / ``"max"``. ``"per_token"`` returns
        ``hidden`` unchanged (``(B, S, D)``) — the caller applies ``mask`` itself.

    Rows with an all-zero mask pool to a zero vector rather than raising.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"mode must be one of {POOLING_MODES}, got {mode!r}")
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be (B, S, D), got shape {tuple(hidden.shape)}")
    if tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError(
            f"mask must be (B, S) = {tuple(hidden.shape[:2])}, got {tuple(mask.shape)}"
        )

    if mode == "per_token":
        return hidden

    keep = mask.to(torch.bool)
    m = keep.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)

    if mode == "mean":
        denom = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return (hidden * m).sum(dim=1) / denom

    if mode == "max":
        neg_inf = torch.finfo(hidden.dtype).min
        out = hidden.masked_fill(~keep.unsqueeze(-1), neg_inf).max(dim=1).values
        return out.masked_fill((~keep.any(dim=1)).unsqueeze(-1), 0.0)

    # mode == "last": gather the last kept position per row
    b, s = keep.shape
    positions = torch.arange(s, device=keep.device).expand(b, s)
    last_idx = torch.where(keep, positions, torch.full_like(positions, -1)).max(dim=1).values
    gathered = hidden[torch.arange(b, device=hidden.device), last_idx.clamp(min=0)]
    return gathered.masked_fill((last_idx < 0).unsqueeze(-1), 0.0)
