from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RewardResult:
    name: str
    value: float
    details: Dict[str, Any]
