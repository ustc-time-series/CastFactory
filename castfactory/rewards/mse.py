from __future__ import annotations

import numpy as np

from castfactory.rewards.base import RewardResult


class MSEReward:
    def __init__(self, temperature: float = 1.0, name: str = "mse"):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.name = name

    def compute(self, pred, target) -> RewardResult:
        pred_array = np.asarray(pred, dtype=float)
        target_array = np.asarray(target, dtype=float)
        mse = float(np.mean((pred_array - target_array) ** 2))
        scaled = mse / self.temperature
        inverse_exp = np.exp(-scaled)
        value = float(inverse_exp / (1.0 + inverse_exp))
        return RewardResult(
            name=self.name,
            value=value,
            details={"mse": mse, "temperature": self.temperature},
        )
