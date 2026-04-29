from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.evaluation.metrics import METRIC_REGISTRY


@dataclass
class EvaluationResult:
    metrics: dict
    predictions: List[dict]


class StandardEvaluator:
    def __init__(self, metrics: Iterable[str] = ("mae", "mse")):
        self.metric_names = list(metrics)
        unknown = [name for name in self.metric_names if name not in METRIC_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")

    def evaluate(
        self,
        samples: Iterable[ForecastSample],
        predictor: Callable[[List[ForecastSample]], List[np.ndarray]],
    ) -> EvaluationResult:
        sample_list = list(samples)
        forecasts = predictor(sample_list)
        if len(forecasts) != len(sample_list):
            raise ValueError("predictor must return one forecast per sample")

        pred_arrays = [np.asarray(forecast, dtype=float) for forecast in forecasts]
        target_arrays = [sample.future_unknown_window.values for sample in sample_list]
        all_pred = np.concatenate(pred_arrays, axis=0)
        all_target = np.concatenate(target_arrays, axis=0)
        insample = np.concatenate([sample.observed_window.values for sample in sample_list], axis=0)

        metrics = {}
        for name in self.metric_names:
            metric_fn = METRIC_REGISTRY[name]
            metrics[name] = metric_fn(all_pred, all_target, insample)

        rows = []
        for sample, forecast in zip(sample_list, pred_arrays):
            sample_id = sample.metadata.get("sample_id", str(len(rows)))
            for step_index in range(sample.prediction_length):
                for channel_index, channel_name in enumerate(sample.future_unknown_window.channel_names):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "cutoff_time": str(sample.cutoff_time),
                            "channel": channel_name,
                            "step": step_index + 1,
                            "pred": float(forecast[step_index, channel_index]),
                            "target": float(
                                sample.future_unknown_window.values[step_index, channel_index]
                            ),
                        }
                    )
        return EvaluationResult(metrics=metrics, predictions=rows)
