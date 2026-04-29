from __future__ import annotations

import json
from typing import Iterable, List

from castfactory.data.records import ForecastSample
from castfactory.representation.base import RepresentationAdapter


class SFTDataset:
    def __init__(
        self,
        samples: Iterable[ForecastSample],
        representation: RepresentationAdapter,
        instruction_template: str = "Predict the next {prediction_length} steps.",
    ):
        self.samples: List[ForecastSample] = list(samples)
        self.representation = representation
        self.instruction_template = instruction_template

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        model_input = self.representation.encode(sample)
        instruction = self.instruction_template.format(
            prediction_length=sample.prediction_length,
            cutoff_time=sample.cutoff_time,
        )
        input_parts = [instruction]
        if model_input.text_prompt:
            input_parts.append(model_input.text_prompt)
        forecast = sample.future_unknown_window.values
        if forecast.shape[1] == 1:
            forecast_payload = [float(value) for value in forecast[:, 0]]
        else:
            forecast_payload = forecast.tolist()
        return {
            "input": "\n".join(input_parts),
            "output": json.dumps({"forecast": forecast_payload}),
            "metadata": dict(sample.metadata),
        }
