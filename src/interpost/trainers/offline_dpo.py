"""OfflineDPOTrainer — TRL ``DPOTrainer`` + an activation-derived loss term
(interpost Mode 2, offline).

Phase 2 shape: one hooked layer, pooled completion activations, a user-supplied
differentiable term added to the DPO loss. The `Signal` / `Intervention` ABCs
generalize this in Phase 4.

    def term_fn(chosen_pooled, rejected_pooled):        # each (B, d_model)
        return torch.relu(probe_logit(rejected_pooled))  # penalize "harmful" reps

    trainer = OfflineDPOTrainer(
        model=..., args=DPOConfig(...), train_dataset=...,
        signal_layer=9, signal_term_fn=term_fn, signal_weight=0.1,
    )
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from trl import DPOTrainer

from interpost.activations import HookManager, pool

# (chosen_pooled, rejected_pooled) -> per-sequence quantity to penalize; interpost
# adds ``signal_weight * term.mean()`` to the DPO loss.
SignalTermFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class OfflineDPOTrainer(DPOTrainer):
    """``DPOTrainer`` with an optional activation-derived auxiliary loss term.

    Extra kwargs (all optional — with none set, this is a plain ``DPOTrainer``):
        signal_layer:    decoder-layer index to read residual-stream activations at.
        signal_term_fn:  ``(chosen_pooled, rejected_pooled) -> Tensor``, differentiable.
        signal_weight:   scalar multiplier on ``term.mean()``.
        signal_pooling:  ``"mean" | "last" | "max"`` over completion tokens.
    """

    def __init__(
        self,
        *args,
        signal_layer: int | None = None,
        signal_term_fn: SignalTermFn | None = None,
        signal_weight: float = 1.0,
        signal_pooling: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.signal_layer = signal_layer
        self.signal_term_fn = signal_term_fn
        self.signal_weight = signal_weight
        self.signal_pooling = signal_pooling
        self._hooks: HookManager | None = None
        self._signal_enabled = signal_term_fn is not None and signal_layer is not None

    def _hookmgr(self, model) -> HookManager:
        if self._hooks is None:
            self._hooks = HookManager(self.accelerator.unwrap_model(model))
        return self._hooks

    def _record(self, key: str, value: torch.Tensor) -> None:
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault(key, []).append(float(value.detach()))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._signal_enabled:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        hooks = self._hookmgr(model)
        # first=True: the DPO step forwards twice (policy, then a no-grad reference).
        # Keep the first — the grad-carrying policy activations.
        with hooks.capture([self.signal_layer], first=True) as acts:
            out = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        loss = out[0] if isinstance(out, tuple) else out

        h = acts[self.signal_layer]  # (2B, S, d)
        cmask = inputs["completion_mask"]  # (2B, S), 1 on completion tokens
        b = h.shape[0] // 2  # batch is [chosen ; rejected]
        chosen = pool(h[:b].float(), cmask[:b], self.signal_pooling)  # (B, d)
        rejected = pool(h[b:].float(), cmask[b:], self.signal_pooling)  # (B, d)

        term = self.signal_weight * self.signal_term_fn(chosen, rejected).mean()
        self._record("signal/term", term)
        loss = loss + term.to(loss.dtype)

        return (loss, out[1]) if isinstance(out, tuple) else loss
