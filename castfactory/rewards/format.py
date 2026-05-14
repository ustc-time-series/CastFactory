from __future__ import annotations

from castfactory.parsers import ParseResult
from castfactory.rewards.base import RewardResult


class FormatReward:
    def __init__(
        self,
        prediction_length: int,
        num_channels: int,
        name: str = "format",
        invalid_value: float = -1.0,
    ):
        self.prediction_length = prediction_length
        self.num_channels = num_channels
        self.name = name
        self.invalid_value = float(invalid_value)

    def compute(self, parsed: ParseResult, structure_checks: dict | None = None) -> RewardResult:
        forecast = parsed.point_forecast
        shape_valid = (
            forecast is not None
            and tuple(forecast.shape) == (self.prediction_length, self.num_channels)
        )
        structure_checks = dict(structure_checks or {})
        structure_valid = all(bool(value) for value in structure_checks.values())
        valid = bool(parsed.success and shape_valid and not parsed.fallback_used and structure_valid)
        return RewardResult(
            name=self.name,
            value=1.0 if valid else self.invalid_value,
            details={
                "parse_success": parsed.success,
                "fallback_used": parsed.fallback_used,
                "shape_valid": shape_valid,
                **structure_checks,
            },
        )
