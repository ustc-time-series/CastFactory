from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List

import numpy as np

from castfactory.data.records import ForecastResult, ForecastSample
from castfactory.evaluation.metrics import LLM_METRIC_REGISTRY, METRIC_REGISTRY


@dataclass
class EvaluationResult:
    metrics: dict
    predictions: List[dict]
    metadata: dict = field(default_factory=dict)


class StandardEvaluator:
    protocol_name = "standard"
    training_required = True

    def __init__(self, metrics: Iterable[str] = ("mae", "mse")):
        self.metric_names = list(metrics)
        known = set(METRIC_REGISTRY) | set(LLM_METRIC_REGISTRY)
        unknown = [name for name in self.metric_names if name not in known]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")

    def evaluate(
        self,
        samples: Iterable[ForecastSample],
        predictor: Callable[[List[ForecastSample]], List[np.ndarray]],
    ) -> EvaluationResult:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("StandardEvaluator requires at least one evaluation sample")
        forecasts = predictor(sample_list)
        if len(forecasts) != len(sample_list):
            raise ValueError("predictor must return one forecast per sample")

        pred_arrays, trace_rows = self._validated_forecasts(sample_list, forecasts)
        target_arrays = [sample.future_unknown_window.values for sample in sample_list]
        all_pred = np.concatenate(pred_arrays, axis=0)
        all_target = np.concatenate(target_arrays, axis=0)
        insample = np.concatenate([sample.observed_window.values for sample in sample_list], axis=0)

        rows = self._prediction_rows(sample_list, pred_arrays, trace_rows)
        metrics = self._compute_metrics(all_pred, all_target, insample, rows)
        return EvaluationResult(
            metrics=metrics,
            predictions=rows,
            metadata=self._metadata(),
        )

    def _validated_forecasts(self, sample_list, forecasts):
        pred_arrays = []
        trace_rows = []
        for sample, forecast in zip(sample_list, forecasts):
            pred_array, trace = self._extract_forecast(forecast)
            expected_shape = sample.future_unknown_window.values.shape
            if pred_array.shape != expected_shape:
                raise ValueError(
                    "forecast shape "
                    f"{pred_array.shape} does not match target shape {expected_shape}"
                )
            pred_arrays.append(pred_array)
            trace_rows.append(trace)
        return pred_arrays, trace_rows

    def _extract_forecast(self, forecast):
        if isinstance(forecast, ForecastResult):
            return (
                np.asarray(forecast.point_forecast, dtype=float),
                {
                    "parse_success": forecast.parse_success,
                    "fallback_used": forecast.fallback_used,
                    "raw_response": forecast.raw_response,
                },
            )
        return (
            np.asarray(forecast, dtype=float),
            {"parse_success": True, "fallback_used": False, "raw_response": ""},
        )

    def _compute_metrics(self, pred, target, insample, rows) -> dict:
        metrics = {}
        for name in self.metric_names:
            if name in LLM_METRIC_REGISTRY:
                metrics[name] = LLM_METRIC_REGISTRY[name](rows)
            else:
                metric_fn = METRIC_REGISTRY[name]
                metrics[name] = metric_fn(pred, target, insample)
        return metrics

    def _prediction_rows(self, sample_list, pred_arrays, trace_rows) -> List[dict]:
        rows = []
        for sample, forecast, trace in zip(sample_list, pred_arrays, trace_rows):
            sample_id = sample.metadata.get("sample_id", str(len(rows)))
            for step_index in range(sample.prediction_length):
                channel_names = sample.future_unknown_window.channel_names
                for channel_index, channel_name in enumerate(channel_names):
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
                            "parse_success": bool(trace["parse_success"]),
                            "fallback_used": bool(trace["fallback_used"]),
                            "raw_response": trace["raw_response"],
                        }
                    )
        return rows

    def _metadata(self) -> dict:
        return {
            "protocol": self.protocol_name,
            "training_required": self.training_required,
        }


class RollingEvaluator(StandardEvaluator):
    protocol_name = "rolling"

    def evaluate(
        self,
        samples: Iterable[ForecastSample],
        predictor: Callable[[List[ForecastSample]], List[np.ndarray]],
    ) -> EvaluationResult:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("RollingEvaluator requires at least one evaluation sample")
        forecasts = predictor(sample_list)
        if len(forecasts) != len(sample_list):
            raise ValueError("predictor must return one forecast per sample")
        pred_arrays, trace_rows = self._validated_forecasts(sample_list, forecasts)

        metrics = {}
        for name in self.metric_names:
            if name in LLM_METRIC_REGISTRY:
                rows = self._prediction_rows(sample_list, pred_arrays, trace_rows)
                metrics[name] = LLM_METRIC_REGISTRY[name](rows)
            else:
                metric_fn = METRIC_REGISTRY[name]
                per_origin = [
                    metric_fn(
                        forecast,
                        sample.future_unknown_window.values,
                        sample.observed_window.values,
                    )
                    for sample, forecast in zip(sample_list, pred_arrays)
                ]
                metrics[name] = float(np.mean(per_origin))
        return EvaluationResult(
            metrics=metrics,
            predictions=self._prediction_rows(sample_list, pred_arrays, trace_rows),
            metadata=self._metadata(),
        )


class ZeroShotEvaluator(StandardEvaluator):
    protocol_name = "zero_shot"
    training_required = False
