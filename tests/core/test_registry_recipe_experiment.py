import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class RegistryRecipeExperimentTests(unittest.TestCase):
    def test_registry_registers_and_builds_components(self):
        from castfactory.core.registry import Registry

        class Component:
            def __init__(self, value):
                self.value = value

        registry = Registry("component")
        registry.register("dummy_component", Component)

        built = registry.build({"name": "dummy_component", "value": 7})

        self.assertIsInstance(built, Component)
        self.assertEqual(built.value, 7)

    def test_registry_rejects_duplicate_names(self):
        from castfactory.core.registry import Registry, RegistryError

        registry = Registry("component")
        registry.register("duplicate", object)

        with self.assertRaisesRegex(RegistryError, "already registered"):
            registry.register("duplicate", object)

    def test_registry_exports_all_extension_helpers(self):
        from castfactory.core import registry

        for name in [
            "register_backbone",
            "register_bridge",
            "register_head",
            "register_trainer",
            "register_metric",
            "register_reward",
            "register_objective",
            "register_protocol",
        ]:
            self.assertTrue(hasattr(registry, name), name)

    def test_package_declares_training_extras(self):
        pyproject = Path("pyproject.toml")
        text = pyproject.read_text(encoding="utf-8")

        self.assertIn("hf = [", text)
        self.assertIn('"transformers"', text)
        self.assertIn("peft = [", text)
        self.assertIn('"peft"', text)
        self.assertIn("train = [", text)
        self.assertIn('"accelerate>=0.26.0"', text)

    def test_recipe_loads_yaml_and_applies_defaults(self):
        from castfactory.core.recipe import RecipeConfig

        recipe_text = """
experiment:
  name: unit_recipe
  stage: sft
data:
  reader:
    name: csv
    path: data.csv
  split:
    type: timestamp
    train_end: "2022-01-31 23:00"
    val_end: "2022-02-28 23:00"
    test_end: "2022-03-31 23:00"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_text(recipe_text)

            recipe = RecipeConfig.from_file(path)

        self.assertEqual(recipe.experiment["name"], "unit_recipe")
        self.assertEqual(recipe.experiment["seed"], 42)
        self.assertEqual(recipe.trace["save_recipe"], True)

    def test_recipe_preserves_single_turn_rollout_workflow_section(self):
        from castfactory.core.recipe import RecipeConfig

        recipe = RecipeConfig.from_mapping(
            {
                "experiment": {"name": "single_turn", "stage": "rlvr"},
                "rollout": {
                    "workflow": {
                        "name": "single_turn",
                    }
                },
            }
        )

        self.assertEqual(recipe.rollout["workflow"]["name"], "single_turn")

    def test_recipe_allows_time_series_agent_rollout_workflow(self):
        from castfactory.core.recipe import RecipeConfig

        recipe = RecipeConfig.from_mapping(
            {
                "experiment": {"name": "agentic", "stage": "rlvr"},
                "rollout": {
                    "workflow": {
                        "name": "time_series_agent",
                        "max_steps": 3,
                    }
                },
            }
        )

        self.assertEqual(recipe.rollout["workflow"]["name"], "time_series_agent")
        self.assertEqual(recipe.rollout["workflow"]["max_steps"], 3)

    def test_recipe_rejects_unknown_rlvr_workflow(self):
        from castfactory.core.recipe import RecipeConfig

        with self.assertRaisesRegex(ValueError, "Unknown RLVR rollout workflow"):
            RecipeConfig.from_mapping(
                {
                    "experiment": {"name": "multiturn", "stage": "rlvr"},
                    "rollout": {
                        "workflow": {
                            "name": "trend_then_forecast",
                            "max_turns": 2,
                        }
                    },
                }
            )

    def test_recipe_rejects_unknown_stage(self):
        from castfactory import Experiment

        with self.assertRaisesRegex(ValueError, "recipe.experiment.stage"):
            Experiment.from_mapping(
                {"experiment": {"name": "bad_stage", "stage": "agentic"}}
            )

    def test_experiment_cpt_stage_does_not_fall_through_to_sft(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "cpt_stage", "stage": "cpt"},
                "data": {
                    "reader": {"name": "csv", "path": "missing.csv"},
                    "split": {"type": "ratio", "ratios": [0.7, 0.1, 0.2]},
                },
            }
        )
        experiment._fit_cpt = lambda: {"stage": "cpt"}

        result = experiment.fit()

        self.assertEqual(result["stage"], "cpt")

    def test_experiment_fit_uses_cpt_stage_and_registered_trainer(self):
        from castfactory import Experiment
        from castfactory.core import registry

        class DummyBackbone:
            def __init__(self, model_name):
                self.model = f"model:{model_name}"
                self.tokenizer = "tokenizer"

            def load(self):
                return self

        class DummyTrainer:
            def __init__(self, train_dataset, checkpoint_dir, model, tokenizer):
                self.train_dataset = train_dataset
                self.checkpoint_dir = checkpoint_dir
                self.model = model
                self.tokenizer = tokenizer

            def fit(self):
                return {
                    "status": "trained",
                    "num_examples": len(self.train_dataset),
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "text": self.train_dataset[0]["text"],
                }

        try:
            registry.register_backbone("dummy_cpt_backbone", DummyBackbone)
            registry.register_trainer("dummy_cpt", DummyTrainer)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                csv_path = tmp_path / "series.csv"
                csv_path.write_text(
                    "date,OT\n"
                    "2022-01-01 00:00,1.0\n"
                    "2022-01-01 01:00,2.0\n"
                    "2022-01-01 02:00,3.0\n"
                    "2022-01-01 03:00,4.0\n"
                    "2022-01-01 04:00,5.0\n",
                    encoding="utf-8",
                )
                experiment = Experiment.from_mapping(
                    {
                        "experiment": {"name": "cpt_fit", "stage": "cpt"},
                        "data": {
                            "reader": {
                                "name": "csv",
                                "path": str(csv_path),
                                "timestamp_col": "date",
                                "target_channels": ["OT"],
                            },
                            "split": {
                                "type": "timestamp",
                                "train_end": "2022-01-01 02:00",
                                "val_end": "2022-01-01 03:00",
                                "test_end": "2022-01-01 04:00",
                            },
                            "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                        },
                        "model": {
                            "backbone": {
                                "name": "dummy_cpt_backbone",
                                "model_name": "tiny",
                            }
                        },
                        "training": {
                            "trainer": {"name": "dummy_cpt"},
                            "checkpoint_dir": str(tmp_path / "checkpoints"),
                        },
                    }
                )

                result = experiment.fit()
        finally:
            registry.clear_all()

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["num_examples"], 1)
        self.assertEqual(result["model"], "model:tiny")
        self.assertIn("<channel> OT", result["text"])

    def test_experiment_from_recipe_preserves_resolved_config(self):
        from castfactory import Experiment

        recipe_text = """
experiment:
  name: unit_experiment
  stage: sft
data:
  reader:
    name: csv
    path: data.csv
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_text(recipe_text)

            experiment = Experiment.from_recipe(path)

        self.assertEqual(experiment.name, "unit_experiment")
        self.assertEqual(experiment.recipe.experiment["stage"], "sft")

    def test_experiment_model_source_prefers_training_init_checkpoint(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "model_source"},
                "model": {"backbone": {"name": "hf_causal_lm", "model_name": "base-model"}},
                "training": {"init_checkpoint": "checkpoints/cpt"},
            }
        )

        config = experiment._resolve_backbone_config()

        self.assertEqual(config["model_name"], "checkpoints/cpt")

    def test_experiment_model_source_rejects_invalid_local_init_checkpoint(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "empty_checkpoint"
            checkpoint_dir.mkdir()
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "invalid_checkpoint"},
                    "model": {"backbone": {"name": "hf_causal_lm", "model_name": "base-model"}},
                    "training": {"init_checkpoint": str(checkpoint_dir)},
                }
            )

            with self.assertRaisesRegex(ValueError, "training.init_checkpoint.*config.json"):
                experiment._resolve_backbone_config()

    def test_experiment_model_source_uses_backbone_without_init_checkpoint(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "model_source_base"},
                "model": {"backbone": {"name": "hf_causal_lm", "model_name": "base-model"}},
            }
        )

        config = experiment._resolve_backbone_config()

        self.assertEqual(config["model_name"], "base-model")

    def test_experiment_model_source_allows_disabled_backbone_section(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "model_source_disabled"},
                "model": {"backbone": None},
            }
        )

        config = experiment._resolve_backbone_config()

        self.assertEqual(config, {})

    def test_experiment_evaluate_wires_data_to_run_store(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
                "2022-01-01 05:00,6.0\n"
                "2022-01-01 06:00,7.0\n"
            )
            recipe_path = tmp_path / "recipe.yaml"
            recipe_path.write_text(
                f"""
experiment:
  name: eval_smoke
data:
  reader:
    name: csv
    path: {csv_path}
    timestamp_col: date
    target_channels: [OT]
  split:
    type: timestamp
    train_end: "2022-01-01 01:00"
    val_end: "2022-01-01 03:00"
    test_end: "2022-01-01 06:00"
  window:
    context_length: 2
    prediction_length: 1
    stride: 1
evaluation:
  split: test
  metrics: [mae, mse]
trace:
  run_root: {tmp_path / "runs"}
"""
            )

            experiment = Experiment.from_recipe(recipe_path)
            result = experiment.evaluate()
            report_path = experiment.report()

            predictions = pd.read_parquet(tmp_path / "runs" / "eval_smoke" / "predictions.parquet")

        self.assertIn("mae", result["metrics"])
        self.assertEqual(len(predictions), 1)
        self.assertTrue(report_path.name.endswith("report.md"))

    def test_experiment_evaluate_supports_protocols_list(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
                "2022-01-01 05:00,6.0\n"
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "eval_protocols"},
                    "data": {
                        "reader": {
                            "name": "csv",
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 01:00",
                            "val_end": "2022-01-01 03:00",
                            "test_end": "2022-01-01 05:00",
                        },
                        "window": {"context_length": 1, "prediction_length": 1},
                    },
                    "evaluation": {"protocols": ["standard", "rolling"], "metrics": ["mae"]},
                    "trace": {"run_root": str(tmp_path / "runs")},
                }
            )

            result = experiment.evaluate()

        self.assertIn("standard", result["protocols"])
        self.assertIn("rolling", result["protocols"])
        self.assertIn("standard.mae", result["metrics"])

    def test_experiment_report_uses_existing_metrics(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "report_existing"},
                    "trace": {"run_root": tmp},
                }
            )
            store = experiment._run_store()
            store.write_json("metrics.json", {"mae": 0.25})

            report_path = experiment.report()
            text = report_path.read_text(encoding="utf-8")

        self.assertIn("0.25", text)

    def test_experiment_predict_uses_last_value_baseline(self):
        from castfactory import Experiment

        recipe = {
            "experiment": {"name": "predict_smoke"},
            "data": {"window": {"prediction_length": 3}},
        }
        experiment = Experiment.from_mapping(recipe)

        result = experiment.predict({"context_values": [[1.0], [2.5]]})

        self.assertEqual(result["forecast"], [[2.5], [2.5], [2.5]])

    def test_experiment_rejects_unknown_evaluation_protocol(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping({"experiment": {"name": "bad_protocol"}})

        with self.assertRaisesRegex(ValueError, "Unknown evaluation protocol"):
            experiment._build_evaluator("agentic", ["mae"])

    def test_experiment_builds_hybrid_representation_from_recipe(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "hybrid_recipe"},
                "representation": {
                    "name": "hybrid",
                    "components": [
                        {"name": "context", "include_domain": True},
                        {"name": "statistics", "features": ["mean"]},
                    ],
                },
            }
        )

        representation = experiment._build_representation()

        self.assertEqual(len(representation.components), 2)

    def test_experiment_fit_can_delegate_to_registered_trainer(self):
        from castfactory import Experiment
        from castfactory.core import registry

        class DummyBackbone:
            def __init__(self, model_name):
                self.model_name = model_name
                self.model = None
                self.tokenizer = None

            def load(self):
                self.model = f"model:{self.model_name}"
                self.tokenizer = f"tokenizer:{self.model_name}"
                return self

        class DummyTrainer:
            def __init__(self, marker, train_dataset, checkpoint_dir, model, tokenizer):
                self.marker = marker
                self.train_dataset = train_dataset
                self.checkpoint_dir = checkpoint_dir
                self.model = model
                self.tokenizer = tokenizer

            def fit(self):
                return {
                    "status": "trained",
                    "marker": self.marker,
                    "num_examples": len(self.train_dataset),
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                }

        try:
            registry.register_backbone("dummy_hf", DummyBackbone)
            registry.register_trainer("dummy", DummyTrainer)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                csv_path = tmp_path / "series.csv"
                csv_path.write_text(
                    "date,OT\n"
                    "2022-01-01 00:00,1.0\n"
                    "2022-01-01 01:00,2.0\n"
                    "2022-01-01 02:00,3.0\n"
                    "2022-01-01 03:00,4.0\n"
                    "2022-01-01 04:00,5.0\n"
                    "2022-01-01 05:00,6.0\n"
                )
                experiment = Experiment.from_mapping(
                    {
                        "experiment": {"name": "fit_delegate"},
                        "data": {
                            "reader": {
                                "name": "csv",
                                "path": str(csv_path),
                                "timestamp_col": "date",
                                "target_channels": ["OT"],
                            },
                            "split": {
                                "type": "timestamp",
                                "train_end": "2022-01-01 03:00",
                                "val_end": "2022-01-01 04:00",
                                "test_end": "2022-01-01 05:00",
                            },
                            "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                        },
                        "model": {"backbone": {"name": "dummy_hf", "model_name": "tiny"}},
                        "representation": {"name": "statistics", "features": ["mean", "last"]},
                        "training": {
                            "trainer": {"name": "dummy", "marker": "ok"},
                            "checkpoint_dir": str(tmp_path / "checkpoints"),
                        },
                    }
                )

                result = experiment.fit()
        finally:
            registry.clear_all()

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["marker"], "ok")
        self.assertEqual(result["num_examples"], 2)
        self.assertTrue(result["checkpoint_dir"].endswith("checkpoints"))
        self.assertEqual(result["model"], "model:tiny")
        self.assertEqual(result["tokenizer"], "tokenizer:tiny")

    def test_experiment_fit_builds_default_sft_dataset_from_recipe(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT,hour\n"
                "2022-01-01 00:00,1.0,0\n"
                "2022-01-01 01:00,2.0,1\n"
                "2022-01-01 02:00,3.0,2\n"
                "2022-01-01 03:00,4.0,3\n"
                "2022-01-01 04:00,5.0,4\n"
                "2022-01-01 05:00,6.0,5\n"
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "default_fit"},
                    "data": {
                        "reader": {
                            "name": "csv",
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                            "covariate_channels": ["hour"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 03:00",
                            "val_end": "2022-01-01 04:00",
                            "test_end": "2022-01-01 05:00",
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "representation": {"name": "statistics", "features": ["mean", "last"]},
                    "training": {"checkpoint_dir": str(tmp_path / "checkpoints")},
                }
            )

            result = experiment.fit()

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "No SFT backend configured")
        self.assertEqual(result["num_examples"], 2)
        self.assertTrue(result["checkpoint_dir"].endswith("checkpoints"))

    def test_experiment_fit_sft_does_not_load_existing_output_dir_without_model(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "stage_metadata.json").write_text("{}", encoding="utf-8")
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n",
                encoding="utf-8",
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "sft_existing_output"},
                    "data": {
                        "reader": {
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 02:00",
                            "val_end": "2022-01-01 03:00",
                            "test_end": "2022-01-01 04:00",
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "training": {"checkpoint_dir": str(checkpoint_dir)},
                }
            )

            result = experiment.fit()

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "No SFT backend configured")

    def test_experiment_fit_cpt_does_not_load_existing_output_dir_without_model(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "stage_metadata.json").write_text("{}", encoding="utf-8")
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n",
                encoding="utf-8",
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "cpt_existing_output", "stage": "cpt"},
                    "data": {
                        "reader": {
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 02:00",
                            "val_end": "2022-01-01 03:00",
                            "test_end": "2022-01-01 04:00",
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "training": {"checkpoint_dir": str(checkpoint_dir)},
                }
            )

            result = experiment.fit()

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "No CPT backend configured")

    def test_experiment_fit_applies_peft_adapter(self):
        from castfactory import Experiment
        from castfactory.core import registry

        class DummyBackbone:
            def __init__(self, model_name):
                self.model_name = model_name
                self.model = f"model:{model_name}"
                self.tokenizer = "tokenizer"

            def load(self):
                return self

        class DummyPEFTAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def apply(self, model):
                return f"peft({model})"

        class DummyTrainer:
            def __init__(self, train_dataset, checkpoint_dir, model, tokenizer):
                self.model = model

            def fit(self):
                return {"model": self.model}

        try:
            registry.register_backbone("dummy_peft_backbone", DummyBackbone)
            registry.register_trainer("dummy_peft_trainer", DummyTrainer)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                csv_path = tmp_path / "series.csv"
                csv_path.write_text(
                    "date,OT\n"
                    "2022-01-01 00:00,1.0\n"
                    "2022-01-01 01:00,2.0\n"
                    "2022-01-01 02:00,3.0\n"
                    "2022-01-01 03:00,4.0\n"
                    "2022-01-01 04:00,5.0\n"
                )
                experiment = Experiment.from_mapping(
                    {
                        "experiment": {"name": "fit_peft"},
                        "data": {
                            "reader": {
                                "name": "csv",
                                "path": str(csv_path),
                                "timestamp_col": "date",
                                "target_channels": ["OT"],
                            },
                            "split": {
                                "type": "timestamp",
                                "train_end": "2022-01-01 02:00",
                                "val_end": "2022-01-01 03:00",
                                "test_end": "2022-01-01 04:00",
                            },
                            "window": {"context_length": 1, "prediction_length": 1},
                        },
                        "model": {
                            "backbone": {
                                "name": "dummy_peft_backbone",
                                "model_name": "tiny",
                            },
                            "peft": {"method": "lora", "r": 4},
                        },
                        "training": {"trainer": {"name": "dummy_peft_trainer"}},
                    }
                )
                with patch("castfactory.core.experiment.PEFTAdapter", DummyPEFTAdapter):
                    result = experiment.fit()
        finally:
            registry.clear_all()

        self.assertEqual(result["model"], "peft(model:tiny)")

    def test_experiment_builds_fsdp_backend_with_long_context_defaults(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "fsdp_backend"},
                "training": {"backend": {"name": "fsdp"}},
            }
        )

        backend = experiment._build_training_backend()

        self.assertEqual(backend.max_prompt_length, 4096)
        self.assertEqual(backend.max_response_length, 8192)
        self.assertEqual(backend.max_length, 12288)
        self.assertEqual(backend.default_train_args["fsdp"], "full_shard auto_wrap")
        self.assertEqual(
            backend.default_train_args["fsdp_config"]["state_dict_type"],
            "SHARDED_STATE_DICT",
        )

    def test_experiment_fsdp_backend_allows_overrides(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "fsdp_backend_overrides"},
                "training": {
                    "backend": {
                        "name": "fsdp",
                        "max_response_length": 4096,
                        "fsdp_config": {
                            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
                        },
                    },
                },
            }
        )

        backend = experiment._build_training_backend()

        self.assertEqual(backend.max_response_length, 4096)
        self.assertEqual(
            backend.default_train_args["fsdp_config"]["state_dict_type"],
            "SHARDED_STATE_DICT",
        )
        self.assertEqual(
            backend.default_train_args["fsdp_config"]["transformer_layer_cls_to_wrap"],
            ["Qwen2DecoderLayer"],
        )

    def test_experiment_cpt_fsdp_backend_uses_distributed_defaults(self):
        from castfactory import Experiment

        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "cpt_fsdp_backend", "stage": "cpt"},
                "training": {"backend": {"name": "fsdp"}},
            }
        )

        backend = experiment._build_cpt_backend()

        self.assertEqual(backend.max_length, 12288)
        self.assertEqual(backend.default_train_args["fsdp"], "full_shard auto_wrap")
        self.assertEqual(
            backend.default_train_args["fsdp_config"]["state_dict_type"],
            "SHARDED_STATE_DICT",
        )

    def test_experiment_fit_uses_rlvr_stage_and_prepares_verl_artifacts(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "rlvr_fit", "stage": "rlvr"},
                    "data": {
                        "reader": {
                            "name": "csv",
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 02:00",
                            "val_end": "2022-01-01 03:00",
                            "test_end": "2022-01-01 04:00",
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "training": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "rewards": [{"name": "format"}],
                    },
                }
            )

            result = experiment.fit()

        self.assertIn(result["status"], {"prepared", "skipped"})
        self.assertTrue(result["dataset_path"].endswith("rollout_dataset.jsonl"))
        self.assertTrue(result["config_path"].endswith("verl_config.yaml"))

    def test_experiment_fit_supports_rlvr_ratio_split_recipe(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
                "2022-01-01 05:00,6.0\n"
                "2022-01-01 06:00,7.0\n"
                "2022-01-01 07:00,8.0\n"
                "2022-01-01 08:00,9.0\n"
                "2022-01-01 09:00,10.0\n"
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "rlvr_ratio_fit", "stage": "rlvr"},
                    "data": {
                        "reader": {
                            "name": "csv",
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "ratio",
                            "ratios": [0.7, 0.1, 0.2],
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "representation": {"name": "markdown_table", "significant_digits": 2},
                    "training": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "rewards": [
                            {"name": "format"},
                            {"name": "mse", "temperature": 1.0},
                        ],
                    },
                }
            )

            result = experiment.fit()

        self.assertEqual(result["status"], "prepared")
        self.assertTrue(result["dataset_path"].endswith("rollout_dataset.jsonl"))

    def test_experiment_fit_rlvr_can_load_instruction_template_file(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
                "2022-01-01 05:00,6.0\n"
                "2022-01-01 06:00,7.0\n"
                "2022-01-01 07:00,8.0\n"
                "2022-01-01 08:00,9.0\n"
                "2022-01-01 09:00,10.0\n"
            )
            template_path = tmp_path / "prompt.txt"
            template_path.write_text(
                "Dataset {dataset_name}; attr {attr_meaning}; look_back {look_back}; "
                "pred_window {pred_window}\n{data_lookback}",
                encoding="utf-8",
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "rlvr_prompt_file", "stage": "rlvr"},
                    "data": {
                        "reader": {
                            "name": "csv",
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                            "static_context": {
                                "dataset_name": "ToySet",
                                "attr_meaning": "load",
                            },
                        },
                        "split": {
                            "type": "ratio",
                            "ratios": [0.7, 0.1, 0.2],
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "representation": {"name": "markdown_table", "significant_digits": 2},
                    "training": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "instruction_template_file": str(template_path),
                        "rewards": [{"name": "format"}],
                    },
                }
            )

            result = experiment.fit()
            lines = (tmp_path / "checkpoints" / "rlvr" / "rollout_dataset.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result["status"], "prepared")
        self.assertIn("Dataset ToySet; attr load; look_back 2; pred_window 1", lines)

    def test_experiment_fit_rlvr_template_file_can_infer_dataset_name_and_attr(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "mini_series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
                "2022-01-01 05:00,6.0\n"
                "2022-01-01 06:00,7.0\n"
                "2022-01-01 07:00,8.0\n"
                "2022-01-01 08:00,9.0\n"
                "2022-01-01 09:00,10.0\n"
            )
            template_path = tmp_path / "prompt.txt"
            template_path.write_text(
                "Dataset {dataset_name}; attr {attr_meaning}\n{data_lookback}",
                encoding="utf-8",
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "rlvr_prompt_file_infer", "stage": "rlvr"},
                    "data": {
                        "reader": {
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "ratio",
                            "ratios": [0.7, 0.1, 0.2],
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "representation": {"name": "markdown_table", "significant_digits": 2},
                    "training": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "instruction_template_file": str(template_path),
                        "rewards": [{"name": "format"}],
                    },
                }
            )

            result = experiment.fit()
            lines = (tmp_path / "checkpoints" / "rlvr" / "rollout_dataset.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result["status"], "prepared")
        self.assertIn("Dataset mini_series; attr OT", lines)

    def test_experiment_fit_rlvr_passes_backend_verl_tree_into_generated_config(self):
        from castfactory import Experiment

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "series.csv"
            csv_path.write_text(
                "date,OT\n"
                "2022-01-01 00:00,1.0\n"
                "2022-01-01 01:00,2.0\n"
                "2022-01-01 02:00,3.0\n"
                "2022-01-01 03:00,4.0\n"
                "2022-01-01 04:00,5.0\n"
            )
            experiment = Experiment.from_mapping(
                {
                    "experiment": {"name": "rlvr_verl_tree", "stage": "rlvr"},
                    "data": {
                        "reader": {
                            "path": str(csv_path),
                            "timestamp_col": "date",
                            "target_channels": ["OT"],
                        },
                        "split": {
                            "type": "timestamp",
                            "train_end": "2022-01-01 02:00",
                            "val_end": "2022-01-01 03:00",
                            "test_end": "2022-01-01 04:00",
                        },
                        "window": {"context_length": 2, "prediction_length": 1, "stride": 1},
                    },
                    "training": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "init_checkpoint": "./checkpoints/demo",
                        "rewards": [{"name": "format"}],
                        "backend": {
                            "algorithm": "grpo",
                            "verl": {
                                "trainer": {"total_epochs": 5},
                                "rollout": {"temperature": 0.7},
                            },
                        },
                    },
                }
            )

            result = experiment.fit()
            config_text = (
                tmp_path / "checkpoints" / "rlvr" / "verl_config.yaml"
            ).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "prepared")
        self.assertIn("total_epochs: 5", config_text)
        self.assertIn("temperature: 0.7", config_text)

    def test_experiment_rlvr_rows_include_model_input_text_without_workflow_context(self):
        import numpy as np
        import pandas as pd

        from castfactory import Experiment
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0], [2.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={"dataset_name": "ToySet", "attr_meaning": "load"},
            metadata={},
        )
        future = TSRecord(
            values=np.array([[3.0]]),
            timestamps=pd.date_range("2022-01-01 02:00", periods=1, freq="h"),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=1,
            metadata={"sample_id": "row-1"},
        )
        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "rlvr_rows", "stage": "rlvr"},
                "representation": {"name": "markdown_table", "significant_digits": 1},
                "training": {"instruction_template": "Forecast {pred_window}\n{data_lookback}"},
            }
        )

        row = experiment._build_rlvr_rows([sample])[0]

        self.assertNotIn("workflow", row)
        self.assertIn("| timestamp | OT |", row["model_input_text"])
        self.assertIn("Forecast 1", row["instruction"])

    def test_experiment_agentic_rlvr_rows_export_raw_chat_and_ground_truth_string(self):
        import numpy as np
        import pandas as pd

        from castfactory import Experiment
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0], [2.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={"dataset_name": "ToySet", "attr_meaning": "load"},
            metadata={},
        )
        future = TSRecord(
            values=np.array([[3.0], [4.0]]),
            timestamps=pd.date_range("2022-01-01 02:00", periods=2, freq="h"),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=2,
            metadata={"sample_id": "row-1"},
        )
        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "agentic_rows", "stage": "rlvr"},
                "rollout": {"workflow": {"name": "time_series_agent"}},
            }
        )

        row = experiment._build_rlvr_rows([sample])[0]

        self.assertEqual(row["agent_name"], "time_series_forecast_agent")
        self.assertIsInstance(row["prompt"], list)
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertIn("2022-01-01 00:00:00 1.000", row["prompt"][0]["content"])
        self.assertEqual(row["reward_model"]["style"], "rule")
        self.assertIn("2022-01-01 02:00:00 3.000", row["reward_model"]["ground_truth"])
        self.assertEqual(row["extra_info"]["prediction_length"], 2)
        self.assertEqual(row["extra_info"]["channel_names"], ["OT"])
        self.assertEqual(row["extra_info"]["label"], [[3.0], [4.0]])

    def test_experiment_agentic_rlvr_rejects_multichannel_samples(self):
        import numpy as np
        import pandas as pd

        from castfactory import Experiment
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0, 10.0], [2.0, 20.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["OT", "HUFL"],
            target_channels=["OT", "HUFL"],
            covariate_channels=[],
            metadata={},
        )
        future = TSRecord(
            values=np.array([[3.0, 30.0]]),
            timestamps=pd.date_range("2022-01-01 02:00", periods=1, freq="h"),
            channel_names=["OT", "HUFL"],
            target_channels=["OT", "HUFL"],
            covariate_channels=[],
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=1,
        )
        experiment = Experiment.from_mapping(
            {
                "experiment": {"name": "agentic_multichannel", "stage": "rlvr"},
                "rollout": {"workflow": {"name": "time_series_agent"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "time_series_agent.*univariate"):
            experiment._build_rlvr_rows([sample])


if __name__ == "__main__":
    unittest.main()
