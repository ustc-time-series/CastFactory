from __future__ import annotations

from typing import Iterable, Mapping

from castfactory.rewards.base import RewardResult


class CompositeReward:
    def __init__(self, weights: Mapping[str, float]):
        self.weights = dict(weights)

    def combine(self, rewards: Iterable[RewardResult]) -> RewardResult:
        details = {}
        total = 0.0
        for reward in rewards:
            details[reward.name] = reward.value
            total += self.weights.get(reward.name, 0.0) * reward.value
        return RewardResult(name="composite", value=float(total), details=details)
