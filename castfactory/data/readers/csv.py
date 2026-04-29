from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from castfactory.data.records import TSRecord


class CSVReader:
    def __init__(
        self,
        path: str | Path,
        timestamp_col: str = "date",
        target_channels: Optional[Iterable[str]] = None,
        covariate_channels: Optional[Iterable[str]] = None,
        static_context: Optional[Dict[str, Any]] = None,
    ):
        self.path = Path(path)
        self.timestamp_col = timestamp_col
        self.target_channels = list(target_channels or [])
        self.covariate_channels = list(covariate_channels or [])
        self.static_context = dict(static_context or {})

    def read(self) -> TSRecord:
        frame = pd.read_csv(self.path)
        if self.timestamp_col not in frame.columns:
            raise ValueError(f"CSV timestamp column '{self.timestamp_col}' not found")
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame[self.timestamp_col]))
        value_columns = [column for column in frame.columns if column != self.timestamp_col]
        if not value_columns:
            raise ValueError("CSVReader requires at least one value column")
        if not self.target_channels:
            self.target_channels = [value_columns[0]]
        missing_targets = set(self.target_channels) - set(value_columns)
        if missing_targets:
            raise ValueError(f"target channels not found in CSV: {sorted(missing_targets)}")
        missing_covariates = set(self.covariate_channels) - set(value_columns)
        if missing_covariates:
            raise ValueError(f"covariate channels not found in CSV: {sorted(missing_covariates)}")
        return TSRecord(
            values=frame[value_columns].to_numpy(dtype=float),
            timestamps=timestamps,
            channel_names=value_columns,
            target_channels=self.target_channels,
            covariate_channels=self.covariate_channels,
            static_context=self.static_context,
            metadata={"source_path": str(self.path)},
        )
