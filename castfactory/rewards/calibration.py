from __future__ import annotations

import numpy as np

from castfactory.rewards.base import RewardResult


class CalibrationReward:
    def __init__(self, lower_key: str = "q10", upper_key: str = "q90", name: str = "calibration"):
        self.lower_key = lower_key
        self.upper_key = upper_key
        self.name = name

    def compute(self, quantile_forecast: dict, target) -> RewardResult:
        if self.lower_key not in quantile_forecast or self.upper_key not in quantile_forecast:
            raise ValueError(
                f"quantile_forecast must include '{self.lower_key}' and '{self.upper_key}'"
            )
        lower = np.asarray(quantile_forecast[self.lower_key], dtype=float)
        upper = np.asarray(quantile_forecast[self.upper_key], dtype=float)
        target_array = np.asarray(target, dtype=float)
        if lower.shape != target_array.shape or upper.shape != target_array.shape:
            raise ValueError("quantile forecast shapes must match target shape")
        covered = (target_array >= lower) & (target_array <= upper)
        coverage = float(np.mean(covered))
        interval_width = float(np.mean(upper - lower))
        # Score centered on full coverage, clipped to reward range.
        value = max(-1.0, min(1.0, 2.0 * coverage - 1.0))
        return RewardResult(
            name=self.name,
            value=value,
            details={"coverage": coverage, "interval_width": interval_width},
        )
