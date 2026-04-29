from __future__ import annotations

from castfactory.parsers import ParseResult
from castfactory.rewards.base import RewardResult


class FormatReward:
    def __init__(self, prediction_length: int, num_channels: int, name: str = "format"):
        self.prediction_length = prediction_length
        self.num_channels = num_channels
        self.name = name

    def compute(self, parsed: ParseResult) -> RewardResult:
        forecast = parsed.point_forecast
        shape_valid = (
            forecast is not None
            and tuple(forecast.shape) == (self.prediction_length, self.num_channels)
        )
        valid = bool(parsed.success and shape_valid and not parsed.fallback_used)
        return RewardResult(
            name=self.name,
            value=1.0 if valid else 0.0,
            details={
                "parse_success": parsed.success,
                "fallback_used": parsed.fallback_used,
                "shape_valid": shape_valid,
            },
        )
