from __future__ import annotations

from typing import List, Optional

from castfactory.data.records import ForecastSample, TSRecord


class WindowBuilder:
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        stride: Optional[int] = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride or prediction_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def build(self, record: TSRecord) -> List[ForecastSample]:
        total = len(record)
        window = self.context_length + self.prediction_length
        samples: List[ForecastSample] = []
        if total < window:
            return samples
        for start in range(0, total - window + 1, self.stride):
            context_stop = start + self.context_length
            target_stop = context_stop + self.prediction_length
            observed = record.slice(start, context_stop)
            future_slice = record.slice(context_stop, target_stop)
            future_unknown = future_slice.select_channels(
                record.target_channels,
                target_channels=record.target_channels,
                covariate_channels=[],
            )
            future_known = None
            if record.covariate_channels:
                future_known = future_slice.select_channels(
                    record.covariate_channels,
                    target_channels=[],
                    covariate_channels=record.covariate_channels,
                )
            samples.append(
                ForecastSample(
                    observed_window=observed,
                    future_known_window=future_known,
                    future_unknown_window=future_unknown,
                    cutoff_time=observed.timestamps[-1],
                    prediction_length=self.prediction_length,
                    metadata={"sample_start": start},
                )
            )
        return samples
