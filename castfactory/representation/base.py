from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np

from castfactory.data.records import ForecastSample


@dataclass
class ModelInput:
    embeddings: Optional[np.ndarray] = None
    token_ids: Optional[np.ndarray] = None
    text_prompt: Optional[str] = None
    metadata: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata or {})


class RepresentationAdapter(Protocol):
    def encode(self, sample: ForecastSample) -> ModelInput:
        ...
