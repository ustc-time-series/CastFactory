from __future__ import annotations

from typing import Iterable

from castfactory.training.rlvr_dataset import RLVRDataset


class RLVRTrainer:
    def __init__(self, dataset: RLVRDataset, rewards: Iterable, backend):
        self.dataset = dataset
        self.rewards = list(rewards)
        self.backend = backend

    def fit(self) -> dict:
        if self.backend is None:
            return {
                "status": "skipped",
                "reason": "No RLVR backend configured",
                "num_prompts": len(self.dataset),
            }
        if hasattr(self.backend, "run"):
            return dict(self.backend.run(self.dataset.prompts(), self.rewards) or {})
        return dict(self.backend(self.dataset.prompts(), self.rewards) or {})
