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


if __name__ == "__main__":
    unittest.main()
