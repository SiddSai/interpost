"""Trainer layer — thin TRL subclasses that call into Signal / Intervention.

Planned: ``OfflineDPOTrainer`` (Phase 1, subclass of ``trl.DPOTrainer``),
``OnlineGRPOTrainer`` (Phase 5, subclass of ``trl.GRPOTrainer``), ``OnlinePPOTrainer``
(later), and ``RewardModelWrapper`` for the online reward-model intervention surface.
Intentionally empty until Phase 1.
"""

__all__: list[str] = []
