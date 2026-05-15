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
                "num_rows": len(self.dataset),
            }
        if hasattr(self.backend, "fit"):
            return dict(self.backend.fit(self.dataset, self.rewards) or {})
        return dict(self.backend(self.dataset, self.rewards) or {})
