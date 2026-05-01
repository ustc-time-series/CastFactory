from __future__ import annotations

import numpy as np

from castfactory.representation import ModelInput


class ProjectorBridge:
    def __init__(self, weights=None, bias=None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.bias = None if bias is None else np.asarray(bias, dtype=float)

    def project(self, model_input: ModelInput) -> ModelInput:
        if model_input.embeddings is None:
            raise ValueError("ProjectorBridge requires embeddings")
        embeddings = np.asarray(model_input.embeddings, dtype=float)
        projected = embeddings
        if self.weights is not None:
            projected = projected @ self.weights
        if self.bias is not None:
            projected = projected + self.bias
        metadata = dict(model_input.metadata)
        metadata["bridge"] = "projector"
        return ModelInput(
            embeddings=projected,
            token_ids=model_input.token_ids,
            text_prompt=model_input.text_prompt,
            metadata=metadata,
        )
