from __future__ import annotations

from typing import Iterable, List

import numpy as np

from castfactory.data.records import ForecastSample
from castfactory.representation.base import ModelInput, RepresentationAdapter


class HybridRepresentation:
    def __init__(self, components: Iterable[RepresentationAdapter]):
        self.components = list(components)
        if not self.components:
            raise ValueError("HybridRepresentation requires at least one component")

    def encode(self, sample: ForecastSample) -> ModelInput:
        prompts: List[str] = []
        embeddings = []
        token_ids = []
        metadata = {"representation": "hybrid", "components": []}
        for component in self.components:
            model_input = component.encode(sample)
            if model_input.text_prompt:
                prompts.append(model_input.text_prompt)
            if model_input.embeddings is not None:
                embeddings.append(model_input.embeddings)
            if model_input.token_ids is not None:
                token_ids.append(model_input.token_ids)
            metadata["components"].append(component.__class__.__name__)
        merged_embeddings = None
        if embeddings:
            merged_embeddings = np.concatenate(embeddings, axis=0)
        merged_token_ids = None
        if token_ids:
            merged_token_ids = np.concatenate(token_ids, axis=0)
        return ModelInput(
            embeddings=merged_embeddings,
            token_ids=merged_token_ids,
            text_prompt="\n".join(prompts) if prompts else None,
            metadata=metadata,
        )
