from __future__ import annotations

import numpy as np

from castfactory.data.records import TSRecord


class TrainOnlyStandardScaler:
    fit_scope = "train_only"

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train_record: TSRecord) -> "TrainOnlyStandardScaler":
        self.mean_ = train_record.values.mean(axis=0, keepdims=True)
        std = train_record.values.std(axis=0, keepdims=True)
        self.std_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, record: TSRecord) -> TSRecord:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyStandardScaler must be fit before transform")
        return record.with_values((record.values - self.mean_) / self.std_)

    def fit_transform(self, train_record: TSRecord) -> TSRecord:
        return self.fit(train_record).transform(train_record)
