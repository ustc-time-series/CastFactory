from __future__ import annotations

from typing import Any, Dict

from castfactory.training.base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(
        self,
        train_dataset=None,
        checkpoint_dir: str = "checkpoints/sft",
        backend=None,
        model=None,
        tokenizer=None,
        train_args: Dict[str, Any] | None = None,
    ):
        super().__init__(checkpoint_dir=checkpoint_dir)
        self.train_dataset = train_dataset
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.train_args = dict(train_args or {})

    def fit(self) -> Dict[str, Any]:
        if self.train_dataset is None:
            raise ValueError("SFTTrainer.fit requires a train_dataset")
        if self.backend is None:
            return {
                "status": "skipped",
                "reason": "No SFT backend configured",
                "num_examples": len(self.train_dataset),
                "checkpoint_dir": str(self.checkpoint_dir),
            }
        payload = {
            "train_dataset": self.train_dataset,
            "checkpoint_dir": self.checkpoint_dir,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_args": dict(self.train_args),
        }
        if hasattr(self.backend, "fit"):
            result = self.backend.fit(**payload)
        else:
            result = self.backend(**payload)
        return dict(result or {})
