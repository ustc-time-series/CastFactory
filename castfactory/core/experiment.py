from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from castfactory.data.readers import CSVReader
from castfactory.data.leakage import LeakageChecker
from castfactory.data.splits import RatioSplitter, TimestampSplitter
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
    MarkdownTableRepresentation,
    NumericalPatchRepresentation,
    StatisticsRepresentation,
    TextualSummaryRepresentation,
)
from castfactory.trace import RunStore
from castfactory.training import (
    CPTDataset,
    CPTTrainer,
    RLVRDataset,
    RLVRTrainer,
    SFTDataset,
    SFTTrainer,
    TransformersCPTBackend,
    TransformersSFTBackend,
)
from castfactory.training.backends import VerlBackend
from castfactory.rewards import AccuracyReward, CalibrationReward, FormatReward, MSEReward, ReasoningReward
from castfactory.training.prompt_template import (
    build_instruction_format_kwargs,
    load_instruction_template,
    template_uses_data_lookback,
)
from castfactory.training.sft_dataset import DEFAULT_INSTRUCTION_TEMPLATE


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
        stage = str(self.recipe.experiment.get("stage", "sft")).lower()
        if not self.recipe.data:
            return {"status": "skipped", "reason": "No training data configured"}
        if stage == "cpt":
            return self._record_fit_stage(self._fit_cpt(), stage)
        if stage == "rlvr":
            return self._record_fit_stage(self._fit_rlvr(), stage)
        if stage == "sft":
            return self._record_fit_stage(self._fit_sft(), stage)
        raise ValueError(f"Unknown experiment stage '{stage}'")

    def _record_fit_stage(self, result: Dict[str, Any], stage: str) -> Dict[str, Any]:
        store = self._run_store()
        store.save_stage_metadata(
            {
                "stage": stage,
                "status": result.get("status", ""),
                "input_checkpoint": self.recipe.training.get("init_checkpoint", ""),
                "output_checkpoint": result.get(
                    "checkpoint_dir",
                    self.recipe.training.get("checkpoint_dir", ""),
                ),
            }
        )
        return result

    def _fit_sft(self) -> Dict[str, Any]:
        train_split = self.recipe.training.get("split", "train")
        samples = self._build_window_samples(train_split)
        representation = self._build_representation()
        instruction_template = self._resolve_instruction_template()
        dataset_kwargs = {
            "samples": samples,
            "representation": representation,
            "instruction_template": instruction_template,
        }
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

    def _fit_cpt(self) -> Dict[str, Any]:
        train_split = self.recipe.training.get("split", "train")
        samples = self._build_window_samples(train_split)
        dataset = CPTDataset(samples)
        checkpoint_dir = self.recipe.training.get("checkpoint_dir", "checkpoints/cpt")
        backend = self._build_cpt_backend()
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
        trainer = CPTTrainer(
            train_dataset=dataset,
            checkpoint_dir=checkpoint_dir,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            train_args=self.recipe.training.get("args", {}),
        )
        return trainer.fit()

    def _build_cpt_backend(self):
        config = self.recipe.training.get("backend")
        if not config:
            if self.recipe.model.get("backbone") if self.recipe.model else None:
                return TransformersCPTBackend()
            return None
        backend_config = dict(config)
        name = backend_config.pop("name", "transformers")
        if name == "fsdp":
            backend_config = self._merge_training_backend_defaults(
                self._default_cpt_fsdp_backend_config(),
                backend_config,
            )
            return TransformersCPTBackend(**backend_config)
        if name == "transformers":
            return TransformersCPTBackend(**backend_config)
        raise ValueError(
            "Unknown CPT training backend "
            f"'{name}'. Available built-in training backends: transformers, fsdp"
        )

    def _fit_rlvr(self) -> Dict[str, Any]:
        train_split = self.recipe.training.get("split", "train")
        samples = self._build_window_samples(train_split)
        rows = self._build_rlvr_rows(samples)
        dataset = RLVRDataset(rows)
        rewards = self._build_rl_rewards(rows)
        backend = self._build_rl_backend()
        trainer = RLVRTrainer(dataset=dataset, rewards=rewards, backend=backend)
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
        split_type = split_config.get("type", "timestamp")
        if split_type == "timestamp":
            split = TimestampSplitter(
                train_end=split_config["train_end"],
                val_end=split_config["val_end"],
                test_end=split_config["test_end"],
            ).split(record)
        elif split_type == "ratio":
            split = RatioSplitter(ratios=split_config["ratios"]).split(record)
        else:
            raise ValueError(
                "Experiment currently supports only timestamp or ratio split recipes"
            )
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
        if name == "markdown_table":
            return MarkdownTableRepresentation(**config)
        if name == "discrete_token":
            return DiscreteTokenRepresentation(**config)
        raise ValueError(
            "Unknown representation "
            f"'{name}'. Available built-in representations: context, discrete_token, "
            "hybrid, markdown_table, numerical_patch, statistics, textual_summary"
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

    def _build_rlvr_rows(self, samples) -> list[dict]:
        representation = self._build_representation()
        instruction_template = self._resolve_instruction_template()
        rows = []
        for index, sample in enumerate(samples):
            model_input = representation.encode(sample)
            format_kwargs = build_instruction_format_kwargs(sample, model_input)
            instruction = instruction_template.format(**format_kwargs)
            input_parts = [instruction]
            if model_input.text_prompt and not template_uses_data_lookback(instruction_template):
                input_parts.append(model_input.text_prompt)
            if sample.future_known_window is not None:
                input_parts.append(self._format_future_known(sample))
            rows.append(
                {
                    "prompt": "\n".join(input_parts),
                    "label": sample.future_unknown_window.values.tolist(),
                    "prediction_length": sample.prediction_length,
                    "channel_names": list(sample.future_unknown_window.channel_names),
                    "observed_values": sample.observed_window.values.tolist(),
                    "cutoff_time": str(sample.cutoff_time),
                    "sample_id": str(sample.metadata.get("sample_id", index)),
                    "metadata": dict(sample.metadata),
                    "instruction": instruction,
                    "model_input_text": model_input.text_prompt or "",
                }
            )
        return rows

    def _build_rl_rewards(self, rows: list[dict]) -> list[object]:
        reward_configs = list(self.recipe.training.get("rewards", []))
        if not reward_configs:
            raise ValueError("training.rewards is required for rlvr stage")
        first_row = rows[0]
        prediction_length = int(first_row["prediction_length"])
        num_channels = len(first_row["channel_names"])
        built = []
        for config in reward_configs:
            name = str(config.get("name", "")).strip().lower()
            if name == "format":
                built.append(
                    FormatReward(
                        prediction_length=prediction_length,
                        num_channels=num_channels,
                    )
                )
            elif name == "accuracy":
                built.append(AccuracyReward(metric=config.get("metric", "mae")))
            elif name == "calibration":
                built.append(
                    CalibrationReward(
                        lower_key=config.get("lower_key", "q10"),
                        upper_key=config.get("upper_key", "q90"),
                    )
                )
            elif name == "mse":
                built.append(MSEReward(temperature=float(config.get("temperature", 1.0))))
            elif name == "reasoning":
                built.append(ReasoningReward(max_horizon_steps=prediction_length))
            else:
                raise ValueError(f"Unknown rlvr reward '{name}'")
        return built

    def _build_rl_backend(self):
        backend_config = dict(self.recipe.training.get("backend", {}))
        name = str(backend_config.pop("name", "verl"))
        if name != "verl":
            raise ValueError("Experiment currently supports only the 'verl' rl backend")
        checkpoint_dir = self.recipe.training.get("checkpoint_dir", "checkpoints/rlvr")
        execute = bool(backend_config.pop("execute", False))
        python_executable = str(backend_config.pop("python_executable", "python"))
        verl_module = str(backend_config.pop("verl_module", "verl.trainer.main_ppo"))
        verl_config_overrides = dict(backend_config.pop("verl", {}))
        model_path = self.recipe.training.get("init_checkpoint")
        if not model_path:
            backbone = self.recipe.model.get("backbone", {})
            model_path = backbone.get("model_name") or backbone.get("name") or ""
        training_config = {
            "algorithm": backend_config.pop("algorithm", "grpo"),
            "epochs": self.recipe.training.get("epochs", 1),
            "batch_size": self.recipe.training.get("batch_size", 1),
            "num_generations": backend_config.pop("num_generations", 1),
            **backend_config,
        }
        model_config = {
            "model_path": model_path,
            "init_checkpoint": self.recipe.training.get("init_checkpoint", ""),
        }
        return VerlBackend(
            run_dir=checkpoint_dir,
            training_config=training_config,
            model_config=model_config,
            config_overrides=verl_config_overrides,
            python_executable=python_executable,
            verl_module=verl_module,
            execute=execute,
        )

    def _format_future_known(self, sample) -> str:
        assert sample.future_known_window is not None
        payload = {}
        for channel_index, channel_name in enumerate(sample.future_known_window.channel_names):
            payload[channel_name] = [
                float(value) for value in sample.future_known_window.values[:, channel_index]
            ]
        return "Known future covariates: " + json.dumps(payload, sort_keys=True)

    def _resolve_instruction_template(self) -> str:
        template_file = self.recipe.training.get("instruction_template_file")
        if template_file:
            return load_instruction_template(template_file, base_dir=self._recipe_base_dir())
        template = self.recipe.training.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE)
        return load_instruction_template(template, base_dir=self._recipe_base_dir())

    def _recipe_base_dir(self) -> Path:
        if self.recipe_path is not None:
            return self.recipe_path.parent
        return Path.cwd()

    def _build_training_backend(self):
        config = self.recipe.training.get("backend")
        if not config:
            if self.recipe.model.get("backbone") if self.recipe.model else None:
                return TransformersSFTBackend(**self._default_fsdp_backend_config())
            return None
        backend_config = dict(config)
        name = backend_config.pop("name", "transformers")
        if name in {"fsdp", "transformers"}:
            if name == "fsdp":
                backend_config = self._merge_training_backend_defaults(
                    self._default_fsdp_backend_config(),
                    backend_config,
                )
            return TransformersSFTBackend(**backend_config)
        raise ValueError(
            "Unknown training backend "
            f"'{name}'. Available built-in training backends: fsdp, transformers"
        )

    def _default_fsdp_backend_config(self) -> dict:
        return {
            "max_prompt_length": 4096,
            "max_response_length": 8192,
            "max_seq_length": 12288,
            "truncation_side": "left",
            "bf16": True,
            "gradient_checkpointing": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "state_dict_type": "SHARDED_STATE_DICT",
            },
        }

    def _default_cpt_fsdp_backend_config(self) -> dict:
        return {
            "max_length": 12288,
            "bf16": True,
            "gradient_checkpointing": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "state_dict_type": "SHARDED_STATE_DICT",
            },
        }

    def _merge_training_backend_defaults(self, defaults: dict, overrides: dict) -> dict:
        merged = dict(defaults)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_training_backend_defaults(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _build_model_assets(self):
        backbone = self._build_backbone(allow_checkpoint_dir=False)
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

    def _build_backbone(self, allow_checkpoint_dir: bool = True):
        model_config = self._resolve_backbone_config(allow_checkpoint_dir=allow_checkpoint_dir)
        if not model_config:
            return None
        backbone_config = dict(model_config)
        name = backbone_config.pop("name", "hf_causal_lm")
        if name in {"hf", "hf_causal_lm"}:
            backbone = HFCausalLMBackbone(**backbone_config)
        else:
            backbone = backbones.build({"name": name, **backbone_config})
        if hasattr(backbone, "load"):
            backbone = backbone.load()
        return backbone

    def _resolve_backbone_config(self, allow_checkpoint_dir: bool = True) -> dict:
        raw_backbone_config = self.recipe.model.get("backbone", {}) if self.recipe.model else {}
        backbone_config = dict(raw_backbone_config or {})
        init_checkpoint = self.recipe.training.get("init_checkpoint")
        if init_checkpoint:
            init_checkpoint_text = str(init_checkpoint)
            checkpoint_path = Path(init_checkpoint_text).expanduser()
            if self._is_explicit_local_path(init_checkpoint_text) and not checkpoint_path.exists():
                raise FileNotFoundError(
                    "training.init_checkpoint points to a local path that does not exist: "
                    f"{init_checkpoint_text}"
                )
            if checkpoint_path.exists():
                self._validate_local_hf_checkpoint(
                    checkpoint_path,
                    field_name="training.init_checkpoint",
                )
            backbone_config.setdefault("name", "hf_causal_lm")
            backbone_config["model_name"] = init_checkpoint_text
            if checkpoint_path.exists():
                backbone_config.setdefault("local_files_only", True)
        if allow_checkpoint_dir and not backbone_config:
            checkpoint_dir = self.recipe.training.get("checkpoint_dir")
            checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
            if checkpoint_path is not None and checkpoint_path.exists():
                backbone_config = {
                    "name": "hf_causal_lm",
                    "model_name": str(checkpoint_path),
                    "local_files_only": True,
                }
        return backbone_config

    def _is_explicit_local_path(self, value: str) -> bool:
        path = Path(value).expanduser()
        return (
            path.is_absolute()
            or value.startswith(("./", "../", ".\\", "..\\", "~"))
            or "\\" in value
        )

    def _validate_local_hf_checkpoint(self, path: Path, *, field_name: str) -> None:
        if not path.is_dir():
            raise ValueError(
                f"{field_name} points to '{path}', but HuggingFace checkpoints must be "
                "directories containing config.json"
            )
        if not (path / "config.json").is_file():
            raise ValueError(
                f"{field_name} points to local checkpoint '{path}', but it is not a valid "
                "HuggingFace checkpoint directory: missing config.json. Run the upstream "
                f"stage successfully or set {field_name} to a valid model checkpoint."
            )

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
