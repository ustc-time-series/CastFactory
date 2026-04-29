from __future__ import annotations

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class TextualSummaryRepresentation:
    def __init__(self, template: str = "default_ts_summary_v1"):
        self.template = template

    def encode(self, sample: ForecastSample) -> ModelInput:
        values = sample.observed_window.values
        start = sample.observed_window.timestamps[0]
        end = sample.observed_window.timestamps[-1]
        prompt = (
            f"time_series_summary(template={self.template}, start={start}, end={end}, "
            f"steps={len(sample.observed_window)}, channels={sample.observed_window.channel_names}, "
            f"last_values={values[-1].tolist()})"
        )
        return ModelInput(text_prompt=prompt, metadata={"representation": "textual_summary"})
