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
    rollout: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "name" not in self.experiment:
            raise ValueError("recipe.experiment.name is required")
        self.experiment.setdefault("seed", 42)
        self.experiment.setdefault("stage", "sft")
        stage = str(self.experiment["stage"]).lower()
        if stage not in {"cpt", "sft", "rlvr"}:
            raise ValueError("recipe.experiment.stage must be one of: cpt, sft, rlvr")
        self.experiment["stage"] = stage
        self.trace.setdefault("save_recipe", True)
        self.trace.setdefault("save_prompt", True)
        self.trace.setdefault("save_response", True)
        self.trace.setdefault("save_parsed", True)
        self.trace.setdefault("save_artifacts", True)
        workflow = self.rollout.get("workflow")
        if stage == "rlvr" and workflow:
            workflow_name = str(workflow.get("name", "single_turn")).lower()
            supported_workflows = {"single_turn", "time_series_agent"}
            if workflow_name not in supported_workflows:
                raise ValueError(
                    "Unknown RLVR rollout workflow "
                    f"'{workflow_name}'. Available workflows: single_turn, time_series_agent"
                )
            workflow["name"] = workflow_name

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
            rollout=_dict(mapping.get("rollout")),
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
            "rollout": dict(self.rollout),
            "inference": dict(self.inference),
            "evaluation": dict(self.evaluation),
            "trace": dict(self.trace),
        }
