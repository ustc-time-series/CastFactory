from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from castfactory.data.records import ForecastSample


@dataclass
class LeakageIssue:
    sample_index: int
    message: str


class LeakageChecker:
    def check_samples(self, samples: Iterable[ForecastSample]) -> List[LeakageIssue]:
        issues: List[LeakageIssue] = []
        for index, sample in enumerate(samples):
            if sample.observed_window.timestamps[-1] != sample.cutoff_time:
                issues.append(LeakageIssue(index, "observed window must end at cutoff_time"))
            if not (sample.future_unknown_window.timestamps > sample.cutoff_time).all():
                issues.append(LeakageIssue(index, "future_unknown_window must be after cutoff_time"))
            if sample.future_known_window is not None:
                if not (sample.future_known_window.timestamps > sample.cutoff_time).all():
                    issues.append(LeakageIssue(index, "future_known_window must be after cutoff_time"))
        return issues
