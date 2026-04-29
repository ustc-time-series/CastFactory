from __future__ import annotations

from typing import List

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class ContextRepresentation:
    def __init__(self, include_domain: bool = True, include_calendar: bool = False):
        self.include_domain = include_domain
        self.include_calendar = include_calendar

    def encode(self, sample: ForecastSample) -> ModelInput:
        context = sample.observed_window.static_context
        parts: List[str] = []
        if self.include_domain and "domain" in context:
            parts.append(f"domain={context['domain']}")
        if "freq" in context:
            parts.append(f"freq={context['freq']}")
        if self.include_calendar:
            parts.append(f"cutoff_time={sample.cutoff_time}")
            parts.append(f"prediction_length={sample.prediction_length}")
        return ModelInput(text_prompt="context(" + ", ".join(parts) + ")", metadata={"representation": "context"})
