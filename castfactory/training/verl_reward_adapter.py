from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from castfactory.parsers import (
    ParseContext,
    ThinkAnswerForecastParser,
    inspect_reasoning_answer_format,
)
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
