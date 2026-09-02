"""Output-side scorers used for probe *labels* and as the classifier baseline.

These score decoded text (not activations). ``RobertaToxicity`` is the label source
and the classifier baseline for the toxicity reproduction; a Llama-Guard wrapper is
added in Phase 2 for the refusal side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class RobertaToxicity:
    """P(toxic) in [0, 1] for a batch of strings, via a RoBERTa toxicity classifier."""

    def __init__(
        self,
        model_id: str = "s-nlp/roberta_toxicity_classifier",
        *,
        device: str | torch.device | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        if device is not None:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size
        self.max_length = max_length
        # find the "toxic" logit index from the model's label map
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.toxic_idx = next(
            (i for i, name in id2label.items() if "toxic" in name and "non" not in name), 1
        )

    @torch.no_grad()
    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        probs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, self.toxic_idx]
            probs.append(p.float().cpu().numpy())
        return np.concatenate(probs, axis=0)

    def labels(self, texts: Sequence[str], threshold: float = 0.5) -> np.ndarray:
        return (self(texts) >= threshold).astype(int)
