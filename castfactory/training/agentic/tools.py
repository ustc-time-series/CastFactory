from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


TIMESTAMP_VALUE_LINE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


CAST_R1_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_basic_statistics",
            "description": "Compute basic statistics for the observed univariate series.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_within_channel_dynamics",
            "description": "Analyze trend, local dynamics, volatility, and peaks.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_forecast_residuals",
            "description": "Summarize residuals from a simple local baseline.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_data_quality",
            "description": "Identify missing, non-finite, flat, and duplicated observations.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_event_summary",
            "description": "Summarize abrupt events and high-magnitude segments.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_time_series",
            "description": "Produce a candidate forecast with a configured model or local fallback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "chronos2, arima, patchtst, itransformer, or last_value",
                    }
                },
            },
        },
    },
]


def format_timestamp_value_lines(
    timestamps: Iterable[Any],
    values: Iterable[Any],
    *,
    precision: int = 3,
) -> str:
    return "\n".join(
        f"{_format_timestamp(timestamp)} {float(value):.{precision}f}"
        for timestamp, value in zip(timestamps, _to_1d_values(values))
    )


def parse_time_series_string(text: str) -> tuple[list[str], list[float]]:
    timestamps: list[str] = []
    values: list[float] = []
    for raw_line in str(text).splitlines():
        match = TIMESTAMP_VALUE_LINE.match(raw_line.strip())
        if not match:
            continue
        timestamps.append(_format_timestamp(match.group(1)))
        values.append(float(match.group(2)))
    if not values:
        raise ValueError("no timestamp/value observations found")
    return timestamps, values


def extract_basic_statistics(
    timestamps: Iterable[Any],
    values: Iterable[Any] | None = None,
) -> dict[str, float]:
    if values is None:
        values = timestamps
    series = _finite_values(values)
    return {
        "count": float(series.size),
        "mean": _safe_float(np.mean(series)),
        "std": _safe_float(np.std(series)),
        "min": _safe_float(np.min(series)),
        "max": _safe_float(np.max(series)),
        "median": _safe_float(np.median(series)),
        "first": _safe_float(series[0]),
        "last": _safe_float(series[-1]),
    }


def extract_within_channel_dynamics(
    timestamps: Iterable[Any],
    values: Iterable[Any] | None = None,
) -> dict[str, float]:
    if values is None:
        values = timestamps
    series = _finite_values(values)
    diffs = np.diff(series)
    x = np.arange(series.size, dtype=float)
    slope = float(np.polyfit(x, series, deg=1)[0]) if series.size > 1 else 0.0
    return {
        "trend_slope": _safe_float(slope),
        "mean_change": _safe_float(np.mean(diffs) if diffs.size else 0.0),
        "mean_abs_change": _safe_float(np.mean(np.abs(diffs)) if diffs.size else 0.0),
        "volatility": _safe_float(np.std(diffs) if diffs.size else 0.0),
        "peak_count": float(_count_peaks(series)),
        "direction_changes": float(_count_direction_changes(diffs)),
    }


def extract_forecast_residuals(
    timestamps: Iterable[Any],
    values: Iterable[Any] | None = None,
) -> dict[str, float]:
    if values is None:
        values = timestamps
    series = _finite_values(values)
    if series.size <= 1:
        residuals = np.array([0.0])
    else:
        residuals = series[1:] - series[:-1]
    return {
        "residual_mean": _safe_float(np.mean(residuals)),
        "residual_std": _safe_float(np.std(residuals)),
        "residual_abs_mean": _safe_float(np.mean(np.abs(residuals))),
        "recent_residual": _safe_float(residuals[-1]),
    }


def extract_data_quality(
    timestamps: Iterable[Any],
    values: Iterable[Any] | None = None,
) -> dict[str, float]:
    if values is None:
        values = timestamps
    timestamp_list = list(timestamps)
    raw_values = np.asarray(_to_1d_values(values), dtype=float)
    nonfinite = int(np.sum(~np.isfinite(raw_values)))
    finite = raw_values[np.isfinite(raw_values)]
    duplicate_timestamps = len(timestamp_list) - len(set(map(str, timestamp_list)))
    zero_diffs = int(np.sum(np.diff(finite) == 0.0)) if finite.size > 1 else 0
    return {
        "quality_nonfinite_ratio": _safe_float(nonfinite / max(1, raw_values.size)),
        "quality_dropout_ratio": _safe_float(zero_diffs / max(1, finite.size - 1)),
        "duplicate_timestamp_count": float(max(0, duplicate_timestamps)),
        "finite_count": float(finite.size),
    }


def extract_event_summary(
    timestamps: Iterable[Any],
    values: Iterable[Any] | None = None,
) -> dict[str, Any]:
    if values is None:
        values = timestamps
    timestamp_list = [_format_timestamp(timestamp) for timestamp in timestamps]
    series = _finite_values(values)
    diffs = np.diff(series)
    threshold = max(float(np.std(diffs)) * 2.0, 1e-12) if diffs.size else math.inf
    event_indices = [index + 1 for index, delta in enumerate(diffs) if abs(float(delta)) > threshold]
    events = [
        {
            "timestamp": timestamp_list[index] if index < len(timestamp_list) else str(index),
            "value": _safe_float(series[index]),
        }
        for index in event_indices[:5]
    ]
    return {
        "event_segment_count": float(len(event_indices)),
        "event_threshold": _safe_float(0.0 if math.isinf(threshold) else threshold),
        "events": events,
    }


async def predict_time_series_async(
    timestamps: Iterable[Any],
    values: Iterable[Any],
    prediction_length: int,
    *,
    model_name: str = "chronos2",
    service_url: str | None = "http://localhost:8994",
    http_client: Any | None = None,
    local_fallback: str = "arima_then_last_value",
    timeout: float = 30.0,
) -> dict[str, Any]:
    timestamp_list = [_format_timestamp(timestamp) for timestamp in timestamps]
    value_list = [float(value) for value in _to_1d_values(values)]
    prediction_length = int(prediction_length)
    model_name = str(model_name or "chronos2").lower()
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")

    if service_url and model_name not in {"arima", "last_value"}:
        try:
            response = await _post_forecast_request(
                service_url=service_url,
                http_client=http_client,
                payload={
                    "model_name": model_name,
                    "timestamps": timestamp_list,
                    "values": value_list,
                    "prediction_length": prediction_length,
                },
                timeout=timeout,
            )
            return _normalize_prediction_response(
                response,
                timestamp_list,
                prediction_length,
                model_used=model_name,
                fallback_used=False,
            )
        except Exception:
            pass

    if local_fallback in {"arima", "arima_then_last_value"} or model_name == "arima":
        try:
            forecast_result = await _run_sync(
                _predict_with_arima_sync,
                timestamp_list,
                value_list,
                prediction_length,
            )
            if isinstance(forecast_result, dict):
                return {
                    **forecast_result,
                    "fallback_used": bool(forecast_result.get("fallback_used", model_name != "arima")),
                }
            return {
                "timestamps": _future_timestamp_strings(timestamp_list, prediction_length),
                "values": [_safe_float(value) for value in forecast_result],
                "model_used": "arima",
                "fallback_used": model_name != "arima",
            }
        except Exception:
            if local_fallback == "arima":
                raise

    return _predict_last_value(timestamp_list, value_list, prediction_length)


def predict_time_series(
    timestamps: Iterable[Any],
    values: Iterable[Any],
    prediction_length: int,
    **kwargs: Any,
) -> dict[str, Any]:
    return asyncio.run(
        predict_time_series_async(timestamps, values, prediction_length, **kwargs)
    )


async def _post_forecast_request(
    *,
    service_url: str,
    http_client: Any | None,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    close_client = False
    client = http_client
    if client is None:
        import httpx

        client = httpx.AsyncClient(timeout=timeout)
        close_client = True
    try:
        url = service_url.rstrip("/") + "/predict"
        try:
            response = await client.post(url, json=payload, timeout=timeout)
        except TypeError:
            response = await client.post(url, json=payload)
        if hasattr(response, "raise_for_status"):
            maybe_result = response.raise_for_status()
            if hasattr(maybe_result, "__await__"):
                await maybe_result
        data = response.json()
        if hasattr(data, "__await__"):
            data = await data
        return dict(data)
    finally:
        if close_client and hasattr(client, "aclose"):
            await client.aclose()


def _normalize_prediction_response(
    response: dict[str, Any],
    observed_timestamps: list[str],
    prediction_length: int,
    *,
    model_used: str,
    fallback_used: bool,
) -> dict[str, Any]:
    raw_values = response.get("values", response.get("forecast", []))
    values = _to_1d_values(raw_values)[:prediction_length]
    if len(values) < prediction_length:
        raise ValueError("model service returned too few forecast values")
    timestamps = response.get("timestamps") or _future_timestamp_strings(
        observed_timestamps,
        prediction_length,
    )
    return {
        "timestamps": [_format_timestamp(timestamp) for timestamp in list(timestamps)[:prediction_length]],
        "values": [_safe_float(value) for value in values],
        "model_used": str(response.get("model_used", model_used)),
        "fallback_used": bool(fallback_used),
    }


def _predict_with_arima_sync(
    timestamps: Iterable[Any],
    values: Iterable[float],
    prediction_length: int,
) -> list[float]:
    del timestamps
    series = np.asarray(list(values), dtype=float)
    if series.size < 3:
        raise ValueError("ARIMA fallback requires at least 3 observations")
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(1, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=int(prediction_length))
    return [float(value) for value in forecast]


def _predict_last_value(
    timestamps: list[str],
    values: list[float],
    prediction_length: int,
) -> dict[str, Any]:
    last = float(values[-1]) if values else 0.0
    return {
        "timestamps": _future_timestamp_strings(timestamps, prediction_length),
        "values": [last for _ in range(prediction_length)],
        "model_used": "last_value",
        "fallback_used": True,
    }


async def _run_sync(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def _future_timestamp_strings(timestamps: Iterable[Any], prediction_length: int) -> list[str]:
    timestamp_list = list(timestamps)
    if not timestamp_list:
        return [str(index) for index in range(prediction_length)]
    parsed = pd.DatetimeIndex(pd.to_datetime(timestamp_list))
    if len(parsed) >= 2:
        step = parsed[-1] - parsed[-2]
    else:
        step = pd.Timedelta(days=1)
    return [
        _format_timestamp(parsed[-1] + step * (index + 1))
        for index in range(int(prediction_length))
    ]


def _to_1d_values(values: Iterable[Any]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim == 0:
        return [float(array)]
    if array.ndim == 1:
        return [float(value) for value in array.tolist()]
    return [float(value) for value in array.reshape(array.shape[0], -1)[:, 0].tolist()]


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(_to_1d_values(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.array([0.0], dtype=float)
    return finite


def _count_peaks(series: np.ndarray) -> int:
    if series.size < 3:
        return 0
    count = 0
    for index in range(1, series.size - 1):
        if series[index] > series[index - 1] and series[index] > series[index + 1]:
            count += 1
    return count


def _count_direction_changes(diffs: np.ndarray) -> int:
    if diffs.size < 2:
        return 0
    signs = np.sign(diffs)
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def _format_timestamp(timestamp: Any) -> str:
    try:
        return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp).replace("T", " ")[:19]


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number
