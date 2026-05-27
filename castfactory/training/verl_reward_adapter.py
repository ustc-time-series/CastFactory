from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from castfactory.parsers import (
    ParseContext,
    TimestampValueForecastParser,
    ThinkAnswerForecastParser,
    inspect_reasoning_answer_format,
)
from castfactory.parsers.timestamp_value_parser import TIMESTAMP_VALUE_PATTERN
from castfactory.rewards import (
    AccuracyReward,
    CalibrationReward,
    FormatReward,
    MSEReward,
    ReasoningReward,
)
from castfactory.rewards.base import RewardResult


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_specs=None,
):
    if _is_agentic_payload(data_source, ground_truth, reward_specs or []):
        return _compute_agentic_score(
            response=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_specs=reward_specs or [],
        )
    row = _row_from_verl_payload(ground_truth=ground_truth, extra_info=extra_info)
    rewards = _build_rewards(reward_specs or [], row)
    result = score_response(
        response=solution_str,
        row=row,
        rewards=rewards,
    )
    result["score"] = result.pop("reward")
    details = result.pop("details", {})
    if isinstance(details, dict):
        for key, value in details.items():
            if isinstance(value, (bool, int, float)):
                result[str(key)] = value
            elif value == "skipped":
                result[str(key)] = -1.0
    return result


def _compute_agentic_score(
    *,
    response: str,
    ground_truth,
    extra_info=None,
    reward_specs=None,
) -> dict:
    extra_info = dict(extra_info or {})
    ground_truth_text = _decode_ground_truth(ground_truth)
    if not isinstance(ground_truth_text, str):
        ground_truth_text = str(ground_truth_text)
    reward_specs = list(reward_specs or []) or _default_agentic_reward_specs()
    component_names = [str(spec.get("name", "")).lower() for spec in reward_specs]
    prediction_length = int(
        extra_info.get("prediction_length")
        or max(1, len(_timestamp_value_rows(ground_truth_text)))
    )
    channel_names = list(extra_info.get("channel_names") or ["value"])
    num_channels = len(channel_names)
    context = ParseContext(
        prediction_length=prediction_length,
        num_channels=num_channels,
        output_schema="timestamp_value",
        channel_names=channel_names,
        observed_values=np.asarray(extra_info.get("observed_values", []), dtype=float),
    )
    structure_checks = inspect_reasoning_answer_format(response)
    parsed = TimestampValueForecastParser().parse(response, context)
    target = TimestampValueForecastParser().parse(
        f"<answer>\n{ground_truth_text}\n</answer>",
        context,
    )

    if extra_info.get("workflow_valid") is False:
        return _agentic_penalty_result(
            penalty=float(extra_info.get("workflow_penalty", -0.5)),
            parsed=parsed,
            component_names=component_names,
            workflow_valid=False,
            workflow_violation=1.0,
        )
    if _answer_copies_observed_timestamp(
        structure_checks.get("answer_payload", ""),
        extra_info.get("observed_timestamps", []),
    ):
        return _agentic_penalty_result(
            penalty=-0.5,
            parsed=parsed,
            component_names=component_names,
            workflow_valid=False,
            workflow_violation=1.0,
        )

    format_reward = FormatReward(
        prediction_length=prediction_length,
        num_channels=num_channels,
    ).compute(
        parsed,
        structure_checks={
            "has_think_block": structure_checks["has_think_block"],
            "has_answer_block": structure_checks["has_answer_block"],
        },
    )
    result = {
        "score": 0.0,
        "parse_success": bool(parsed.success),
        "fallback_used": bool(parsed.fallback_used),
        "format": float(format_reward.value),
        "workflow_valid": True,
        "workflow_penalty": 0.0,
        "workflow_violation": 0.0,
    }
    if format_reward.value < 1.0 or not target.success:
        for name in component_names:
            if name != "format":
                result[name] = -1.0
        result["score"] = -1.0
        return _ordered_agentic_result(result, component_names)

    pred = np.asarray(parsed.point_forecast, dtype=float).reshape(prediction_length, num_channels)
    truth = np.asarray(target.point_forecast, dtype=float).reshape(prediction_length, num_channels)
    score = 0.0
    for spec in reward_specs:
        name = str(spec.get("name", "")).lower()
        if name == "format":
            continue
        if name == "length":
            value = _agentic_length_score(parsed, prediction_length)
        elif name == "normalized_mse":
            value = _normalized_mse_score(pred, truth)
        elif name == "change_point":
            value = _change_point_score(pred[:, 0], truth[:, 0])
        elif name == "season_trend":
            value = _season_trend_score(pred[:, 0], truth[:, 0])
        else:
            value = 0.0
        result[name] = float(value)
        score += float(value)
    result["score"] = float(score)
    return _ordered_agentic_result(result, component_names)


def score_response(
    response: str,
    row: dict,
    rewards: Iterable[object],
) -> dict:
    context = ParseContext(
        prediction_length=int(row["prediction_length"]),
        num_channels=len(row["channel_names"]),
        output_schema="think_answer_array",
        channel_names=list(row["channel_names"]),
        observed_values=np.asarray(row["observed_values"], dtype=float),
    )
    structure_checks = inspect_reasoning_answer_format(response)
    parsed = ThinkAnswerForecastParser().parse(response, context)

    format_reward = None
    other_rewards = []
    reward_details = {}
    for reward in rewards:
        if isinstance(reward, FormatReward):
            format_reward = reward.compute(
                parsed,
                structure_checks={
                    "has_think_block": structure_checks["has_think_block"],
                    "has_answer_block": structure_checks["has_answer_block"],
                },
            )
            reward_details[format_reward.name] = format_reward.value
        else:
            other_rewards.append(reward)

    if format_reward is not None and format_reward.value < 1.0:
        for reward in other_rewards:
            reward_details[getattr(reward, "name", reward.__class__.__name__.lower())] = "skipped"
        return {
            "reward": -1.0,
            "parse_success": parsed.success,
            "fallback_used": parsed.fallback_used,
            "details": reward_details,
        }

    computed_rewards = []
    for reward in other_rewards:
        result = _compute_reward(reward, parsed, row, response)
        computed_rewards.append(result)
        reward_details[result.name] = result.value

    combined_reward = _combine_reward_results(format_reward, computed_rewards)
    return {
        "reward": combined_reward,
        "parse_success": parsed.success,
        "fallback_used": parsed.fallback_used,
        "details": reward_details,
    }


def _combine_reward_results(
    format_reward: RewardResult | None,
    reward_results: list[RewardResult],
) -> float:
    if not reward_results:
        if format_reward is None:
            return 0.0
        return float(format_reward.value)
    if len(reward_results) == 1:
        return float(reward_results[0].value)
    return float(np.mean([result.value for result in reward_results]))


def _compute_reward(reward, parsed, row, response: str) -> RewardResult:
    try:
        if isinstance(reward, AccuracyReward):
            return reward.compute(parsed.point_forecast, row["label"])
        if isinstance(reward, MSEReward):
            return reward.compute(parsed.point_forecast, row["label"])
        if isinstance(reward, CalibrationReward):
            quantiles = parsed.quantile_forecast or {}
            return reward.compute(quantiles, row["label"])
        if isinstance(reward, ReasoningReward):
            return reward.compute(response, cutoff_time=row.get("cutoff_time"))
    except Exception as exc:
        return RewardResult(
            name=getattr(reward, "name", reward.__class__.__name__),
            value=-1.0,
            details={"error": str(exc)},
        )
    raise TypeError(f"Unsupported reward type for verl adapter: {reward.__class__.__name__}")


def _row_from_verl_payload(ground_truth, extra_info=None) -> dict:
    extra_info = dict(extra_info or {})
    ground_truth = _decode_ground_truth(ground_truth)
    if isinstance(ground_truth, dict):
        label = ground_truth.get("label", ground_truth)
        ground_truth_context = ground_truth
    else:
        label = ground_truth
        ground_truth_context = {}
    return {
        "label": label,
        "prediction_length": extra_info.get("prediction_length")
        or ground_truth_context.get("prediction_length"),
        "channel_names": extra_info.get("channel_names") or ground_truth_context.get("channel_names"),
        "observed_values": extra_info.get("observed_values") or ground_truth_context.get("observed_values"),
        "cutoff_time": extra_info.get("cutoff_time") or ground_truth_context.get("cutoff_time"),
        "sample_id": extra_info.get("sample_id") or extra_info.get("index"),
    }


def _decode_ground_truth(ground_truth):
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return ground_truth
    return ground_truth


def _build_rewards(reward_specs: Iterable[dict], row: dict) -> list[object]:
    built = []
    prediction_length = int(row["prediction_length"])
    num_channels = len(row["channel_names"])
    for spec in reward_specs:
        config = dict(spec)
        name = str(config.pop("name", "")).lower()
        if name == "format":
            built.append(
                FormatReward(
                    prediction_length=int(config.pop("prediction_length", prediction_length)),
                    num_channels=int(config.pop("num_channels", num_channels)),
                    **config,
                )
            )
        elif name == "accuracy":
            built.append(AccuracyReward(**config))
        elif name == "calibration":
            built.append(CalibrationReward(**config))
        elif name == "mse":
            built.append(MSEReward(**config))
        elif name == "reasoning":
            built.append(ReasoningReward(**config))
        else:
            raise ValueError(f"Unknown verl reward spec '{name}'")
    return built


def _is_agentic_payload(data_source, ground_truth, reward_specs: Iterable[dict]) -> bool:
    if "agentic" in str(data_source).lower():
        return True
    decoded = _decode_ground_truth(ground_truth)
    if isinstance(decoded, str) and TIMESTAMP_VALUE_PATTERN.search(decoded):
        return True
    agentic_rewards = {"length", "normalized_mse", "change_point", "season_trend"}
    return any(str(spec.get("name", "")).lower() in agentic_rewards for spec in reward_specs)


def _default_agentic_reward_specs() -> list[dict]:
    return [
        {"name": "format"},
        {"name": "length"},
        {"name": "normalized_mse"},
        {"name": "change_point"},
        {"name": "season_trend"},
    ]


def _timestamp_value_rows(text: str) -> list[tuple[str, float]]:
    rows = []
    for line in str(text).splitlines():
        match = TIMESTAMP_VALUE_PATTERN.match(line.strip())
        if match:
            rows.append((match.group(0).rsplit(maxsplit=1)[0], float(match.group(1))))
    return rows


def _answer_copies_observed_timestamp(answer_payload: str, observed_timestamps) -> bool:
    observed = {str(timestamp).replace("T", " ")[:19] for timestamp in (observed_timestamps or [])}
    if not observed:
        return False
    for line in str(answer_payload).splitlines():
        match = TIMESTAMP_VALUE_PATTERN.match(line.strip())
        if match:
            timestamp = line.strip().rsplit(maxsplit=1)[0].replace("T", " ")[:19]
            if timestamp in observed:
                return True
    return False


def _agentic_penalty_result(
    *,
    penalty: float,
    parsed,
    component_names: list[str],
    workflow_valid: bool,
    workflow_violation: float,
) -> dict:
    result = {
        "score": float(penalty),
        "parse_success": bool(getattr(parsed, "success", False)),
        "fallback_used": bool(getattr(parsed, "fallback_used", False)),
        "workflow_valid": bool(workflow_valid),
        "workflow_penalty": float(penalty),
        "workflow_violation": float(workflow_violation),
    }
    for name in component_names:
        result[name] = -1.0 if name != "format" else 0.0
    return _ordered_agentic_result(result, component_names)


def _ordered_agentic_result(result: dict, component_names: list[str]) -> dict:
    ordered = {}
    for key in (
        "score",
        "parse_success",
        "fallback_used",
        "format",
        "workflow_valid",
        "workflow_penalty",
        "workflow_violation",
    ):
        if key in result:
            ordered[key] = result[key]
    for name in component_names:
        if name != "format" and name in result:
            ordered[name] = result[name]
    for key, value in result.items():
        if key not in ordered:
            ordered[key] = value
    return _metric_compatible(ordered)


def _agentic_length_score(parsed, prediction_length: int) -> float:
    if not parsed.success or parsed.point_forecast is None:
        return 0.0
    produced = int(np.asarray(parsed.point_forecast).reshape(-1).size)
    return float(0.1 * min(1.0, produced / max(1, prediction_length)))


def _normalized_mse_score(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    scale = float(np.std(target))
    if scale <= 1e-12:
        scale = max(abs(float(np.mean(target))), 1.0)
    mse = float(np.mean(((pred - target) / scale) ** 2))
    return float(0.6 / (1.0 + np.log1p(mse)))


def _change_point_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_points = _turning_points(pred)
    target_points = _turning_points(target)
    if not target_points:
        return 0.2 if not pred_points else 0.0
    matched = 0
    remaining = set(pred_points)
    for point in target_points:
        candidates = [candidate for candidate in remaining if abs(candidate - point) <= 2]
        if candidates:
            matched += 1
            remaining.remove(candidates[0])
    return float(0.2 * matched / max(1, len(target_points)))


def _season_trend_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if pred.size < 2 or target.size < 2:
        return 0.2
    pred_slope = float(np.polyfit(np.arange(pred.size), pred, deg=1)[0])
    target_slope = float(np.polyfit(np.arange(target.size), target, deg=1)[0])
    trend_score = 1.0 if pred_slope == target_slope == 0 else max(
        0.0,
        1.0 - abs(pred_slope - target_slope) / (abs(target_slope) + 1e-6),
    )
    pred_centered = pred - _moving_average(pred)
    target_centered = target - _moving_average(target)
    denom = float(np.linalg.norm(pred_centered) * np.linalg.norm(target_centered))
    corr_score = 1.0 if denom <= 1e-12 else max(0.0, float(np.dot(pred_centered, target_centered) / denom))
    return float(0.2 * (0.5 * trend_score + 0.5 * corr_score))


def _turning_points(values: np.ndarray) -> list[int]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 3:
        return []
    points = []
    for index in range(1, array.size - 1):
        if (array[index] > array[index - 1] and array[index] > array[index + 1]) or (
            array[index] < array[index - 1] and array[index] < array[index + 1]
        ):
            points.append(index)
    return points


def _moving_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    window = min(5, max(1, values.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _metric_compatible(result: dict) -> dict:
    converted = {}
    for key, value in result.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating, np.bool_)):
            converted[key] = bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
    return converted
