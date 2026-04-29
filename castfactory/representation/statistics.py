from __future__ import annotations

from typing import Iterable, List

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class StatisticsRepresentation:
    def __init__(self, features: Iterable[str] | None = None):
        self.features = list(features or ["mean", "std"])

    def encode(self, sample: ForecastSample) -> ModelInput:
        values = sample.observed_window.values
        lines: List[str] = []
        for channel_index, channel_name in enumerate(sample.observed_window.channel_names):
            channel = values[:, channel_index]
            parts = [f"channel={channel_name}"]
            for feature in self.features:
                parts.append(f"{feature}={self._compute(feature, channel):.4f}")
            lines.append("statistics(" + ", ".join(parts) + ")")
        return ModelInput(text_prompt="\n".join(lines), metadata={"representation": "statistics"})

    def _compute(self, feature: str, channel: np.ndarray) -> float:
        if feature == "mean":
            return float(np.mean(channel))
        if feature == "std":
            return float(np.std(channel))
        if feature == "min":
            return float(np.min(channel))
        if feature == "max":
            return float(np.max(channel))
        if feature == "last":
            return float(channel[-1])
        if feature in {"trend", "trend_strength"}:
            if channel.size < 2:
                return 0.0
            return float(channel[-1] - channel[0]) / max(channel.size - 1, 1)
        if feature.startswith("acf"):
            return self._acf_lag1(channel)
        raise ValueError(f"Unsupported statistics feature: {feature}")

    def _acf_lag1(self, channel: np.ndarray) -> float:
        if channel.size < 2:
            return 0.0
        x = channel[:-1] - channel[:-1].mean()
        y = channel[1:] - channel[1:].mean()
        denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if denominator == 0.0:
            return 0.0
        return float(np.sum(x * y) / denominator)
