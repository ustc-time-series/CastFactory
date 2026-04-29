from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("recipe sections must be mappings")
    return dict(value)


@dataclass
class RecipeConfig:
    experiment: Dict[str, Any]
    data: Dict[str, Any] = field(default_factory=dict)
    representation: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "name" not in self.experiment:
            raise ValueError("recipe.experiment.name is required")
        self.experiment.setdefault("seed", 42)
        self.experiment.setdefault("stage", "sft")
        self.trace.setdefault("save_recipe", True)
        self.trace.setdefault("save_prompt", True)
        self.trace.setdefault("save_response", True)
        self.trace.setdefault("save_parsed", True)
        self.trace.setdefault("save_artifacts", True)

    @classmethod
    def from_file(cls, path: str | Path) -> "RecipeConfig":
        recipe_path = Path(path)
        with recipe_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("recipe file must contain a YAML mapping")
        return cls.from_mapping(loaded)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RecipeConfig":
        return cls(
            experiment=_dict(mapping.get("experiment")),
            data=_dict(mapping.get("data")),
            representation=_dict(mapping.get("representation")),
            model=_dict(mapping.get("model")),
            training=_dict(mapping.get("training")),
            inference=_dict(mapping.get("inference")),
            evaluation=_dict(mapping.get("evaluation")),
            trace=_dict(mapping.get("trace")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": dict(self.experiment),
            "data": dict(self.data),
            "representation": dict(self.representation),
            "model": dict(self.model),
            "training": dict(self.training),
            "inference": dict(self.inference),
            "evaluation": dict(self.evaluation),
            "trace": dict(self.trace),
        }
