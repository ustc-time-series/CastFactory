from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _copy_context(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class TSRecord:
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    channel_names: Sequence[str]
    target_channels: Sequence[str]
    covariate_channels: Sequence[str]
    static_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.timestamps = pd.DatetimeIndex(self.timestamps)
        self.channel_names = list(self.channel_names)
        self.target_channels = list(self.target_channels)
        self.covariate_channels = list(self.covariate_channels)
        self.static_context = _copy_context(self.static_context)
        self.metadata = _copy_context(self.metadata)

        if self.values.ndim != 2:
            raise ValueError("TSRecord.values must be a 2D array with shape (T, C)")
        if len(self.timestamps) != self.values.shape[0]:
            raise ValueError("TSRecord timestamps length must match values.shape[0]")
        if len(self.channel_names) != self.values.shape[1]:
            raise ValueError("channel_names length must match values.shape[1]")
        unknown_targets = set(self.target_channels) - set(self.channel_names)
        if unknown_targets:
            raise ValueError(f"target_channels not present in channel_names: {sorted(unknown_targets)}")
        unknown_covariates = set(self.covariate_channels) - set(self.channel_names)
        if unknown_covariates:
            raise ValueError(
                f"covariate_channels not present in channel_names: {sorted(unknown_covariates)}"
            )

    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def num_channels(self) -> int:
        return self.values.shape[1]

    def slice(self, start: int, stop: int) -> "TSRecord":
        return TSRecord(
            values=self.values[start:stop].copy(),
            timestamps=self.timestamps[start:stop],
            channel_names=list(self.channel_names),
            target_channels=list(self.target_channels),
            covariate_channels=list(self.covariate_channels),
            static_context=dict(self.static_context),
            metadata=dict(self.metadata),
        )

    def select_mask(self, mask: np.ndarray) -> "TSRecord":
        return TSRecord(
            values=self.values[mask].copy(),
            timestamps=self.timestamps[mask],
            channel_names=list(self.channel_names),
            target_channels=list(self.target_channels),
            covariate_channels=list(self.covariate_channels),
            static_context=dict(self.static_context),
            metadata=dict(self.metadata),
        )

    def with_values(self, values: np.ndarray) -> "TSRecord":
        return TSRecord(
            values=values,
            timestamps=self.timestamps,
            channel_names=list(self.channel_names),
            target_channels=list(self.target_channels),
            covariate_channels=list(self.covariate_channels),
            static_context=dict(self.static_context),
            metadata=dict(self.metadata),
        )


@dataclass
class ForecastSample:
    observed_window: TSRecord
    future_known_window: Optional[TSRecord]
    future_unknown_window: TSRecord
    cutoff_time: pd.Timestamp
    prediction_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if len(self.future_unknown_window) != self.prediction_length:
            raise ValueError("future_unknown_window length must match prediction_length")
        self.cutoff_time = pd.Timestamp(self.cutoff_time)
        self.metadata = _copy_context(self.metadata)


@dataclass
class ForecastResult:
    point_forecast: np.ndarray
    quantile_forecast: Optional[Dict[str, np.ndarray]] = None
    raw_response: str = ""
    parse_success: bool = True
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.point_forecast = np.asarray(self.point_forecast, dtype=float)
        self.quantile_forecast = self.quantile_forecast or {}
        self.metadata = _copy_context(self.metadata)
