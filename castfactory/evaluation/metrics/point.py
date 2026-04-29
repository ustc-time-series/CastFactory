from __future__ import annotations

from typing import Optional

import numpy as np


def _arrays(pred, target) -> tuple[np.ndarray, np.ndarray]:
    pred_array = np.asarray(pred, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if pred_array.shape != target_array.shape:
        raise ValueError(f"prediction shape {pred_array.shape} != target shape {target_array.shape}")
    return pred_array, target_array


def mae(pred, target, insample: Optional[np.ndarray] = None) -> float:
    pred_array, target_array = _arrays(pred, target)
    return float(np.mean(np.abs(pred_array - target_array)))


def mse(pred, target, insample: Optional[np.ndarray] = None) -> float:
    pred_array, target_array = _arrays(pred, target)
    return float(np.mean((pred_array - target_array) ** 2))


def rmse(pred, target, insample: Optional[np.ndarray] = None) -> float:
    return float(np.sqrt(mse(pred, target)))


def smape(pred, target, insample: Optional[np.ndarray] = None) -> float:
    pred_array, target_array = _arrays(pred, target)
    denominator = np.abs(pred_array) + np.abs(target_array)
    values = np.where(denominator == 0.0, 0.0, 2.0 * np.abs(pred_array - target_array) / denominator)
    return float(np.mean(values))


def mase(pred, target, insample: Optional[np.ndarray] = None) -> float:
    pred_array, target_array = _arrays(pred, target)
    if insample is None:
        raise ValueError("mase requires insample values")
    insample_array = np.asarray(insample, dtype=float)
    if insample_array.shape[0] < 2:
        scale = 1.0
    else:
        scale = float(np.mean(np.abs(np.diff(insample_array, axis=0))))
        if scale == 0.0:
            scale = 1.0
    return float(np.mean(np.abs(pred_array - target_array)) / scale)


METRIC_REGISTRY = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "smape": smape,
    "mase": mase,
}
