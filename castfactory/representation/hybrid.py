from __future__ import annotations

from typing import Iterable, List

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput, RepresentationAdapter


class HybridRepresentation:
    def __init__(self, components: Iterable[RepresentationAdapter]):
        self.components = list(components)
        if not self.components:
            raise ValueError("HybridRepresentation requires at least one component")

    def encode(self, sample: ForecastSample) -> ModelInput:
        prompts: List[str] = []
        metadata = {"representation": "hybrid", "components": []}
        for component in self.components:
            model_input = component.encode(sample)
            if model_input.text_prompt:
                prompts.append(model_input.text_prompt)
            metadata["components"].append(component.__class__.__name__)
        return ModelInput(text_prompt="\n".join(prompts), metadata=metadata)
