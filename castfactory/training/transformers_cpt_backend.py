from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


class TransformersCPTBackend:
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
        prepared = []
        for row in train_dataset:
            ids = self._tokenize_text(str(row["text"]))[: self.max_length]
            prepared.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": list(ids),
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
            raise ValueError("TransformersCPTBackend.fit requires a model")
        self._ensure_pad_token()
        tokenized_dataset = self.prepare_dataset(train_dataset)
        trainer_dataset = self._training_features(tokenized_dataset)
        trainer_cls, training_args_cls, data_collator_cls = self._trainer_types()
        args = dict(self.default_train_args)
        args.update(train_args or {})
        args.setdefault("output_dir", str(checkpoint_dir))
        try:
            training_args = training_args_cls(**args)
        except ImportError as exc:
            if "accelerate" in str(exc):
                raise ImportError(
                    "TransformersCPTBackend requires the CastFactory training extra "
                    "('castfactory[train]') with accelerate>=0.26.0."
                ) from exc
            raise
        data_collator = data_collator_cls(
            tokenizer=self._require_tokenizer(),
            model=model,
            padding=True,
        )
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=trainer_dataset,
            data_collator=data_collator,
        )
        train_result = trainer.train()
        if hasattr(trainer, "save_model"):
            trainer.save_model(str(checkpoint_dir))
        tokenizer = self._require_tokenizer()
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(checkpoint_dir))
        metrics = getattr(train_result, "metrics", {}) or {}
        return {
            "status": "trained",
            "num_examples": len(tokenized_dataset),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": dict(metrics),
        }

    def _training_features(self, tokenized_dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "metadata"}
            for row in tokenized_dataset
        ]

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("TransformersCPTBackend requires a tokenizer")
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
                "TransformersCPTBackend requires the optional 'transformers' dependency"
            ) from exc
        return (
            self.trainer_cls or Trainer,
            self.training_args_cls or TrainingArguments,
            self.data_collator_cls or DataCollatorForSeq2Seq,
        )

    def _tokenize_text(self, text: str) -> list[int]:
        tokenizer = self._require_tokenizer()
        encoded = tokenizer(text, truncation=False)
        return list(encoded["input_ids"])
