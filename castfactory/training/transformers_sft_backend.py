from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


class TransformersSFTBackend:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        trainer_cls=None,
        training_args_cls=None,
        data_collator_cls=None,
        max_length: int = 2048,
        **default_train_args,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_cls = trainer_cls
        self.training_args_cls = training_args_cls
        self.data_collator_cls = data_collator_cls
        self.max_length = max_length
        self.default_train_args = dict(default_train_args)

    def prepare_dataset(self, train_dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tokenizer = self._require_tokenizer()
        prepared = []
        for row in train_dataset:
            prompt = str(row["input"])
            output = str(row["output"])
            eos_token = getattr(tokenizer, "eos_token", "") or ""
            full_text = prompt + output + eos_token
            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = list(encoded["input_ids"])
            labels = list(input_ids)
            masked = min(len(prompt_ids), len(labels))
            labels[:masked] = [-100] * masked
            prepared.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": list(encoded.get("attention_mask", [1] * len(input_ids))),
                    "labels": labels,
                    "metadata": dict(row.get("metadata", {})),
                }
            )
        return prepared

    def fit(
        self,
        train_dataset,
        checkpoint_dir: str | Path,
        model=None,
        tokenizer=None,
        train_args: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        model = model or self.model
        if model is None:
            raise ValueError("TransformersSFTBackend.fit requires a model")
        self._ensure_pad_token()
        tokenized_dataset = self.prepare_dataset(train_dataset)
        trainer_cls, training_args_cls, data_collator_cls = self._trainer_types()
        args = dict(self.default_train_args)
        args.update(train_args or {})
        args.setdefault("output_dir", str(checkpoint_dir))
        training_args = training_args_cls(**args)
        data_collator = data_collator_cls(
            tokenizer=self._require_tokenizer(),
            model=model,
            padding=True,
        )
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        train_result = trainer.train()
        if hasattr(trainer, "save_model"):
            trainer.save_model(str(checkpoint_dir))
        metrics = getattr(train_result, "metrics", {}) or {}
        return {
            "status": "trained",
            "num_examples": len(tokenized_dataset),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": dict(metrics),
        }

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("TransformersSFTBackend requires a tokenizer")
        return self.tokenizer

    def _ensure_pad_token(self) -> None:
        tokenizer = self._require_tokenizer()
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise ValueError("Tokenizer must define pad_token or eos_token")
            tokenizer.pad_token = eos_token

    def _trainer_types(self):
        if (
            self.trainer_cls is not None
            and self.training_args_cls is not None
            and self.data_collator_cls is not None
        ):
            return self.trainer_cls, self.training_args_cls, self.data_collator_cls
        try:
            from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError(
                "TransformersSFTBackend requires the optional 'transformers' dependency"
            ) from exc
        return (
            self.trainer_cls or Trainer,
            self.training_args_cls or TrainingArguments,
            self.data_collator_cls or DataCollatorForSeq2Seq,
        )
