from __future__ import annotations

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class TextualSummaryRepresentation:
    def __init__(self, template: str = "default_ts_summary_v1"):
        self.template = template

    def encode(self, sample: ForecastSample) -> ModelInput:
        values = sample.observed_window.values
        start = sample.observed_window.timestamps[0]
        end = sample.observed_window.timestamps[-1]
        freq = sample.observed_window.static_context.get("freq", "unknown")
        lines = [
            "The time series spans "
            f"{start} to {end} ({len(sample.observed_window)} steps, freq={freq})."
        ]
        for channel_index, channel_name in enumerate(sample.observed_window.channel_names):
            channel = values[:, channel_index]
            trend_per_step = 0.0
            if channel.size > 1:
                trend_per_step = float(channel[-1] - channel[0]) / (channel.size - 1)
            if abs(trend_per_step) < 1e-12:
                direction = "stable"
            elif trend_per_step > 0:
                direction = "upward"
            else:
                direction = "downward"
            lines.append(
                f'Channel "{channel_name}": mean={float(np.mean(channel)):.4f}, '
                f"std={float(np.std(channel)):.4f}, min={float(np.min(channel)):.4f}, "
                f"max={float(np.max(channel)):.4f}."
            )
            lines.append(
                f"Recent trend: {direction} ({trend_per_step:+.4f}/step). "
                f"Last observed value: {float(channel[-1]):.4f}."
            )
        prompt = "\n".join(lines)
        return ModelInput(text_prompt=prompt, metadata={"representation": "textual_summary"})
