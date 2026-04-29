from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from castfactory.core.recipe import RecipeConfig


@dataclass
class Experiment:
    recipe: RecipeConfig
    recipe_path: Path | None = None

    @classmethod
    def from_recipe(cls, path: str | Path) -> "Experiment":
        recipe_path = Path(path)
        return cls(recipe=RecipeConfig.from_file(recipe_path), recipe_path=recipe_path)

    @property
    def name(self) -> str:
        return str(self.recipe.experiment["name"])

    def fit(self) -> None:
        raise NotImplementedError("Training runners are introduced after the Phase 0 scaffold.")

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError("Evaluation runners are introduced after the Phase 0 scaffold.")

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Prediction runners are introduced after the Phase 0 scaffold.")

    def report(self) -> None:
        raise NotImplementedError("Report generation is introduced after the Phase 0 scaffold.")
