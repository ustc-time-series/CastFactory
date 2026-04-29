from __future__ import annotations

from typing import Any, Dict

from castfactory.training.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(self, train_dataset=None, checkpoint_dir: str = "checkpoints/sft"):
        super().__init__(checkpoint_dir=checkpoint_dir)
        self.train_dataset = train_dataset

    def fit(self) -> Dict[str, Any]:
        if self.train_dataset is None:
            raise ValueError("SFTTrainer.fit requires a train_dataset")
        return {"num_examples": len(self.train_dataset), "checkpoint_dir": str(self.checkpoint_dir)}
