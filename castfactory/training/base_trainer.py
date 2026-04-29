from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class BaseTrainer:
    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> Dict[str, Any]:
        raise NotImplementedError
