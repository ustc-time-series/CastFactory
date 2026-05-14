from __future__ import annotations

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput


class MarkdownTableRepresentation:
    def __init__(self, significant_digits: int = 2):
        if significant_digits < 0:
            raise ValueError("significant_digits must be non-negative")
        self.significant_digits = int(significant_digits)

    def encode(self, sample: ForecastSample) -> ModelInput:
        observed = sample.observed_window
        headers = ["timestamp", *observed.channel_names]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---", *(["---:"] * len(observed.channel_names))]) + " |",
        ]
        for row_index, timestamp in enumerate(observed.timestamps):
            values = []
            for channel_index in range(observed.num_channels):
                value = float(observed.values[row_index, channel_index])
                values.append(f"{value:.{self.significant_digits}f}")
            lines.append("| " + " | ".join([str(timestamp), *values]) + " |")
        return ModelInput(
            text_prompt="\n".join(lines),
            metadata={
                "representation": "markdown_table",
                "significant_digits": self.significant_digits,
            },
        )
