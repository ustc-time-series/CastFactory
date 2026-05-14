from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from castfactory.data.records import TSRecord


@dataclass
class DataSplit:
    train: TSRecord
    val: TSRecord
    test: TSRecord


class TimestampSplitter:
    def __init__(self, train_end: str, val_end: str, test_end: str):
        self.train_end = pd.Timestamp(train_end)
        self.val_end = pd.Timestamp(val_end)
        self.test_end = pd.Timestamp(test_end)
        if not self.train_end < self.val_end < self.test_end:
            raise ValueError("split timestamps must satisfy train_end < val_end < test_end")

    def split(self, record: TSRecord) -> DataSplit:
        timestamps = record.timestamps
        train_mask = timestamps <= self.train_end
        val_mask = (timestamps > self.train_end) & (timestamps <= self.val_end)
        test_mask = (timestamps > self.val_end) & (timestamps <= self.test_end)
        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise ValueError("timestamp split produced an empty train, val, or test partition")
        return DataSplit(
            train=record.select_mask(train_mask),
            val=record.select_mask(val_mask),
            test=record.select_mask(test_mask),
        )


class RatioSplitter:
    def __init__(self, ratios: tuple[float, float, float] | list[float]):
        if len(ratios) != 3:
            raise ValueError("ratios must contain exactly three values")
        self.ratios = tuple(float(value) for value in ratios)
        if any(value <= 0.0 for value in self.ratios):
            raise ValueError("ratios must be positive")
        total = sum(self.ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("ratios must sum to 1.0")

    def split(self, record: TSRecord) -> DataSplit:
        total = len(record)
        train_end = int(total * self.ratios[0])
        val_end = train_end + int(total * self.ratios[1])
        train = record.slice(0, train_end)
        val = record.slice(train_end, val_end)
        test = record.slice(val_end, total)
        if not len(train) or not len(val) or not len(test):
            raise ValueError("ratio split produced an empty train, val, or test partition")
        return DataSplit(train=train, val=val, test=test)
