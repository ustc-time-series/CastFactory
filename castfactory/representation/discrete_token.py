from __future__ import annotations

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class DiscreteTokenRepresentation:
    def __init__(
        self,
        num_bins: int,
        value_min: float,
        value_max: float,
        token_offset: int = 0,
    ):
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        self.num_bins = num_bins
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.token_offset = int(token_offset)

    def encode(self, sample: ForecastSample) -> ModelInput:
        values = sample.observed_window.values.reshape(-1)
        scaled = (values - self.value_min) / (self.value_max - self.value_min)
        bins = np.floor(scaled * self.num_bins).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)
        return ModelInput(
            token_ids=bins + self.token_offset,
            metadata={
                "representation": "discrete_token",
                "num_bins": self.num_bins,
                "token_offset": self.token_offset,
            },
        )
