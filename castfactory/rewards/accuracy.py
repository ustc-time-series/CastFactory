from __future__ import annotations

import numpy as np

from castfactory.evaluation.metrics import METRIC_REGISTRY
from castfactory.rewards.base import RewardResult


class AccuracyReward:
    def __init__(self, metric: str = "mae", name: str = "accuracy"):
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Unknown accuracy reward metric: {metric}")
        self.metric = metric
        self.name = name

    def compute(self, pred, target, insample=None) -> RewardResult:
        metric_value = METRIC_REGISTRY[self.metric](
            np.asarray(pred, dtype=float),
            np.asarray(target, dtype=float),
            None if insample is None else np.asarray(insample, dtype=float),
        )
        return RewardResult(
            name=self.name,
            value=-float(metric_value),
            details={"metric": self.metric, "metric_value": float(metric_value)},
        )
