from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from castfactory.data.readers import CSVReader
from castfactory.data.leakage import LeakageChecker
from castfactory.data.splits import TimestampSplitter
from castfactory.data.windows import WindowBuilder
from castfactory.evaluation.protocols import RollingEvaluator, StandardEvaluator, ZeroShotEvaluator
from castfactory.core.recipe import RecipeConfig
from castfactory.core.registry import backbones, trainers
from castfactory.models.adapters import PEFTAdapter
from castfactory.models.backbones import HFCausalLMBackbone
from castfactory.models.bridges import TextConcatBridge
from castfactory.models.heads import TextGenerationHead
from castfactory.parsers import JSONForecastParser, ParseContext
from castfactory.representation import (
    ContextRepresentation,
    DiscreteTokenRepresentation,
    HybridRepresentation,
    NumericalPatchRepresentation,
    StatisticsRepresentation,
    TextualSummaryRepresentation,
)
from castfactory.trace import RunStore
from castfactory.training import SFTDataset, SFTTrainer, TransformersSFTBackend


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
        if not self.recipe.data:
            return {"status": "skipped", "reason": "No training data configured"}
        train_split = self.recipe.training.get("split", "train")
        samples = self._build_window_samples(train_split)
        representation = self._build_representation()
        dataset_kwargs = {
            "samples": samples,
            "representation": representation,
        }
        if "instruction_template" in self.recipe.training:
            dataset_kwargs["instruction_template"] = self.recipe.training["instruction_template"]
        dataset = SFTDataset(**dataset_kwargs)
        checkpoint_dir = self.recipe.training.get("checkpoint_dir", "checkpoints/sft")
        backend = self._build_training_backend()
        model, tokenizer = self._build_model_assets()
        trainer_config = self.recipe.training.get("trainer")
        if trainer_config:
            trainer = trainers.build(
                trainer_config,
                extra_kwargs={
                    "train_dataset": dataset,
                    "checkpoint_dir": checkpoint_dir,
                    "model": model,
                    "tokenizer": tokenizer,
                },
            )
            if not hasattr(trainer, "fit"):
                raise TypeError("Configured trainer must expose a fit() method")
            return dict(trainer.fit() or {})
        trainer = SFTTrainer(
            train_dataset=dataset,
            checkpoint_dir=checkpoint_dir,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            train_args=self.recipe.training.get("args", {}),
        )
        return trainer.fit()

    def evaluate(self) -> Dict[str, Any]:
        samples = self._build_evaluation_samples()
        metrics = self.recipe.evaluation.get("metrics", ["mae", "mse"])
        protocols = self.recipe.evaluation.get("protocols")
        predictor = self._build_predictor()
        store = self._run_store()
        if protocols:
            protocol_results = {}
            flat_metrics = {}
            all_predictions = []
            for protocol in protocols:
                evaluator = self._build_evaluator(protocol, metrics)
                result = evaluator.evaluate(samples, predictor)
                protocol_results[protocol] = {
                    "metrics": result.metrics,
                    "metadata": result.metadata,
                }
                for name, value in result.metrics.items():
                    flat_metrics[f"{protocol}.{name}"] = value
                for row in result.predictions:
                    protocol_row = dict(row)
                    protocol_row["protocol"] = protocol
                    all_predictions.append(protocol_row)
            self._write_evaluation_artifacts(store, flat_metrics, all_predictions)
            return {
                "metrics": flat_metrics,
                "protocols": protocol_results,
                "predictions": all_predictions,
                "run_dir": str(store.path),
            }
        protocol = self.recipe.evaluation.get("protocol", "standard")
        evaluator = self._build_evaluator(protocol, metrics)
        result = evaluator.evaluate(samples, predictor)
        self._write_evaluation_artifacts(store, result.metrics, result.predictions)
        return {
            "metrics": result.metrics,
            "predictions": result.predictions,
            "run_dir": str(store.path),
        }

    def _write_evaluation_artifacts(
        self,
        store: RunStore,
        metrics: dict,
        predictions: list,
    ) -> None:
        if self.recipe.trace.get("save_recipe", True):
            store.write_json("recipe.json", self.recipe.to_dict())
        store.write_json("metrics.json", metrics)
        store.save_predictions(predictions)
        store.write_leaderboard(metrics)
        store.write_report(self.name, metrics)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" in inputs and self.recipe.model.get("backbone"):
            backbone = self._build_backbone()
            parser = JSONForecastParser()
            prediction_length = int(inputs.get("prediction_length", 1))
            num_channels = int(inputs.get("num_channels", 1))
            context = ParseContext(
                prediction_length=prediction_length,
                num_channels=num_channels,
                output_schema="forecast_json_v1",
                channel_names=list(inputs.get("channel_names", ["value"])),
                observed_values=np.asarray(inputs.get("context_values", []), dtype=float),
            )
            result = TextGenerationHead(parser=parser).generate(
                backbone=backbone,
                prompt=str(inputs["prompt"]),
                parse_context=context,
                **self._generation_kwargs(),
            )
            return {
                "forecast": result.point_forecast.tolist(),
                "parse_success": result.parse_success,
                "fallback_used": result.fallback_used,
                "raw_response": result.raw_response,
            }
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
            metrics_path = store.path / "metrics.json"
            metrics = {}
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            store.write_report(self.name, metrics)
        return path

    def _build_evaluation_samples(self):
        split_name = self.recipe.evaluation.get("split", "test")
        return self._build_window_samples(split_name)

    def _build_window_samples(self, split_name: str):
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
        self._check_leakage(samples)
        return samples

    def _build_representation(self):
        config = self.recipe.representation or {"name": "statistics"}
        return self._build_representation_from_config(config)

    def _build_representation_from_config(self, config: Dict[str, Any]):
        config = dict(config)
        name = config.pop("name", "statistics")
        if name == "statistics":
            return StatisticsRepresentation(**config)
        if name == "textual_summary":
            return TextualSummaryRepresentation(**config)
        if name == "context":
            return ContextRepresentation(**config)
        if name == "hybrid":
            components = [
                self._build_representation_from_config(component)
                for component in config.pop("components")
            ]
            return HybridRepresentation(components=components)
        if name == "numerical_patch":
            return NumericalPatchRepresentation(**config)
        if name == "discrete_token":
            return DiscreteTokenRepresentation(**config)
        raise ValueError(
            "Unknown representation "
            f"'{name}'. Available built-in representations: context, discrete_token, "
            "hybrid, numerical_patch, statistics, textual_summary"
        )

    def _check_leakage(self, samples) -> None:
        policy = self.recipe.data.get("leakage_policy", "error")
        if policy == "off":
            return
        issues = LeakageChecker().check_samples(samples)
        if not issues:
            return
        message = "; ".join(f"sample {issue.sample_index}: {issue.message}" for issue in issues)
        if policy == "warn":
            warnings.warn(f"Leakage check found issues: {message}", RuntimeWarning)
            return
        raise ValueError(f"Leakage check found issues: {message}")

    def _build_training_backend(self):
        config = self.recipe.training.get("backend")
        if not config:
            return None
        backend_config = dict(config)
        name = backend_config.pop("name", "transformers")
        if name == "transformers":
            return TransformersSFTBackend(**backend_config)
        raise ValueError(
            "Unknown training backend "
            f"'{name}'. Available built-in training backends: transformers"
        )

    def _build_model_assets(self):
        backbone = self._build_backbone()
        if backbone is None:
            return None, None
        model = getattr(backbone, "model", backbone)
        tokenizer = getattr(backbone, "tokenizer", None)
        peft_config = self.recipe.model.get("peft") if self.recipe.model else None
        if peft_config and model is not None:
            model = PEFTAdapter(**dict(peft_config)).apply(model)
            if hasattr(backbone, "model"):
                backbone.model = model
        return model, tokenizer

    def _build_backbone(self):
        model_config = self.recipe.model.get("backbone") if self.recipe.model else None
        if not model_config:
            checkpoint_dir = self.recipe.training.get("checkpoint_dir")
            checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
            if checkpoint_path is None or not checkpoint_path.exists():
                return None
            model_config = {
                "name": "hf_causal_lm",
                "model_name": str(checkpoint_path),
                "local_files_only": True,
            }
        backbone_config = dict(model_config)
        name = backbone_config.pop("name", "hf_causal_lm")
        if name in {"hf", "hf_causal_lm"}:
            backbone = HFCausalLMBackbone(**backbone_config)
        else:
            backbone = backbones.build({"name": name, **backbone_config})
        if hasattr(backbone, "load"):
            backbone = backbone.load()
        return backbone

    def _build_predictor(self):
        try:
            backbone = self._build_backbone()
        except Exception as exc:
            warnings.warn(
                f"Could not build model predictor ({exc}); falling back to last-value predictor",
                RuntimeWarning,
            )
            return self._last_value_predictor
        if backbone is None:
            warnings.warn(
                "No model checkpoint/backbone configured; using last-value predictor",
                RuntimeWarning,
            )
            return self._last_value_predictor
        representation = self._build_representation()
        bridge = TextConcatBridge(
            system_prompt=self.recipe.inference.get("system_prompt", "")
        )
        parser = JSONForecastParser()
        head = TextGenerationHead(parser=parser)

        def predictor(samples):
            results = []
            for sample in samples:
                instruction = (
                    f"Predict the next {sample.prediction_length} steps. "
                    'Return JSON: {"forecast": [...]}'
                )
                model_input = representation.encode(sample)
                prompt = bridge.build_prompt(model_input, instruction)
                context = ParseContext(
                    prediction_length=sample.prediction_length,
                    num_channels=sample.future_unknown_window.num_channels,
                    output_schema="forecast_json_v1",
                    channel_names=list(sample.future_unknown_window.channel_names),
                    observed_values=sample.observed_window.values,
                )
                results.append(
                    head.generate(
                        backbone=backbone,
                        prompt=prompt,
                        parse_context=context,
                        **self._generation_kwargs(),
                    )
                )
            return results

        return predictor

    def _generation_kwargs(self) -> dict:
        prompt_only_keys = {"system_prompt"}
        return {
            key: value
            for key, value in self.recipe.inference.items()
            if key not in prompt_only_keys
        }

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
