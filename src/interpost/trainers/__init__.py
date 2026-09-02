"""Trainer layer — thin TRL subclasses that call into Signal / Intervention.

``OfflineDPOTrainer`` (subclass of ``trl.DPOTrainer``) is done. Planned:
``OnlineGRPOTrainer`` (subclass of ``trl.GRPOTrainer``), ``OnlinePPOTrainer``, and
``RewardModelWrapper`` for the online reward-model intervention surface.
"""

from interpost.trainers.offline_dpo import OfflineDPOTrainer

__all__ = ["OfflineDPOTrainer"]
