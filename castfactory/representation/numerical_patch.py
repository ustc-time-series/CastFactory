from __future__ import annotations

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class NumericalPatchRepresentation:
    def __init__(self, patch_size: int, stride: int | None = None):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.stride = stride or patch_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def encode(self, sample: ForecastSample) -> ModelInput:
        values = sample.observed_window.values
        if len(values) < self.patch_size:
            patches = values.reshape(1, -1)
        else:
            patches = []
            for start in range(0, len(values) - self.patch_size + 1, self.stride):
                patch = values[start : start + self.patch_size].reshape(-1)
                patches.append(patch)
            patches = np.asarray(patches, dtype=float)
        return ModelInput(
            embeddings=patches,
            metadata={
                "representation": "numerical_patch",
                "patch_size": self.patch_size,
                "stride": self.stride,
            },
        )
