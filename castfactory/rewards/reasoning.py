from __future__ import annotations

import re

import pandas as pd

from castfactory.rewards.base import RewardResult


class ReasoningReward:
    def __init__(self, max_horizon_steps: int | None = None, name: str = "reasoning"):
        self.max_horizon_steps = max_horizon_steps
        self.name = name

    def compute(self, reasoning_text: str, cutoff_time=None) -> RewardResult:
        value = 1.0
        details = {
            "future_timestamp_leak": False,
            "horizon_valid": True,
        }
        if cutoff_time is not None and self._mentions_future_timestamp(reasoning_text, cutoff_time):
            value -= 0.5
            details["future_timestamp_leak"] = True
        horizon = self._extract_horizon(reasoning_text)
        if self.max_horizon_steps is not None and horizon is not None:
            if horizon > self.max_horizon_steps:
                value -= 0.5
                details["horizon_valid"] = False
        return RewardResult(name=self.name, value=max(-1.0, value), details=details)

    def _mentions_future_timestamp(self, text: str, cutoff_time) -> bool:
        cutoff = pd.Timestamp(cutoff_time)
        for match in re.findall(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?", text):
            if pd.Timestamp(match) > cutoff:
                return True
        return False

    def _extract_horizon(self, text: str) -> int | None:
        match = re.search(r"horizon\s*=\s*(\d+)", text)
        if not match:
            return None
        return int(match.group(1))
