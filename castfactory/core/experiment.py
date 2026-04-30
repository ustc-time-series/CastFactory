from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from castfactory.data.readers import CSVReader
from castfactory.data.splits import TimestampSplitter
from castfactory.data.windows import WindowBuilder
from castfactory.evaluation.protocols import RollingEvaluator, StandardEvaluator, ZeroShotEvaluator
from castfactory.core.recipe import RecipeConfig
from castfactory.core.registry import trainers
from castfactory.trace import RunStore


@dataclass
class Experiment:
    recipe: RecipeConfig
    recipe_path: Path | None = None

    @classmethod
    def from_recipe(cls, path: str | Path) -> "Experiment":
        recipe_path = Path(path)
        return cls(recipe=RecipeConfig.from_file(recipe_path), recipe_path=recipe_path)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "Experiment":
        return cls(recipe=RecipeConfig.from_mapping(mapping), recipe_path=None)

    @property
    def name(self) -> str:
        return str(self.recipe.experiment["name"])

    def fit(self) -> Dict[str, Any]:
        trainer_config = self.recipe.training.get("trainer")
        if trainer_config:
            trainer = trainers.build(trainer_config)
            if not hasattr(trainer, "fit"):
                raise TypeError("Configured trainer must expose a fit() method")
            return dict(trainer.fit() or {})
        return {"status": "skipped", "reason": "No training backend configured"}

    def evaluate(self) -> Dict[str, Any]:
        samples = self._build_evaluation_samples()
        protocol = self.recipe.evaluation.get("protocol", "standard")
        metrics = self.recipe.evaluation.get("metrics", ["mae", "mse"])
        evaluator = self._build_evaluator(protocol, metrics)
        result = evaluator.evaluate(samples, self._last_value_predictor)
        store = self._run_store()
        if self.recipe.trace.get("save_recipe", True):
            store.write_json("recipe.json", self.recipe.to_dict())
        store.write_json("metrics.json", result.metrics)
        store.save_predictions(result.predictions)
        store.write_leaderboard(result.metrics)
        store.write_report(self.name, result.metrics)
        return {
            "metrics": result.metrics,
            "predictions": result.predictions,
            "run_dir": str(store.path),
        }

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        values = np.asarray(inputs["context_values"], dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        prediction_length = int(
            inputs.get(
                "prediction_length",
                self.recipe.data.get("window", {}).get("prediction_length", 1),
            )
        )
        last = values[-1]
        forecast = np.tile(last, (prediction_length, 1))
        return {"forecast": forecast.tolist()}

    def report(self) -> Path:
        store = self._run_store()
        path = store.path / "report.md"
        if not path.exists():
            store.write_report(self.name, {})
        return path

    def _build_evaluation_samples(self):
        data = self.recipe.data
        reader_config = data.get("reader", {})
        if reader_config.get("name", "csv") != "csv":
            raise ValueError("Experiment currently supports only CSV reader recipes")
        reader_kwargs = {key: value for key, value in reader_config.items() if key != "name"}
        record = CSVReader(**reader_kwargs).read()
        split_config = data.get("split", {})
        if split_config.get("type", "timestamp") != "timestamp":
            raise ValueError("Experiment currently supports only timestamp split recipes")
        split = TimestampSplitter(
            train_end=split_config["train_end"],
            val_end=split_config["val_end"],
            test_end=split_config["test_end"],
        ).split(record)
        split_name = self.recipe.evaluation.get("split", "test")
        split_record = getattr(split, split_name)
        window = data.get("window", {})
        builder = WindowBuilder(
            context_length=int(window.get("context_length", 1)),
            prediction_length=int(window.get("prediction_length", 1)),
            stride=int(window.get("stride", window.get("prediction_length", 1))),
        )
        samples = builder.build(split_record)
        for index, sample in enumerate(samples):
            sample.metadata.setdefault("sample_id", str(index))
        return samples

    def _build_evaluator(self, protocol: str, metrics: List[str]):
        if protocol == "rolling":
            return RollingEvaluator(metrics=metrics)
        if protocol == "zero_shot":
            return ZeroShotEvaluator(metrics=metrics)
        if protocol == "standard":
            return StandardEvaluator(metrics=metrics)
        raise ValueError(
            "Unknown evaluation protocol "
            f"'{protocol}'. Available protocols: rolling, standard, zero_shot"
        )

    def _last_value_predictor(self, samples):
        forecasts = []
        for sample in samples:
            target_channels = sample.future_unknown_window.channel_names
            indices = [
                sample.observed_window.channel_names.index(channel) for channel in target_channels
            ]
            last = sample.observed_window.values[-1, indices]
            forecasts.append(np.tile(last, (sample.prediction_length, 1)))
        return forecasts

    def _run_store(self) -> RunStore:
        root = self.recipe.trace.get("run_root", "runs")
        return RunStore(root=root, run_id=self.name)
