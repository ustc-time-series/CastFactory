import tempfile
import unittest
from pathlib import Path

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

    def test_experiment_predict_uses_last_value_baseline(self):
        from castfactory import Experiment

        recipe = {
            "experiment": {"name": "predict_smoke"},
            "data": {"window": {"prediction_length": 3}},
        }
        experiment = Experiment.from_mapping(recipe)

        result = experiment.predict({"context_values": [[1.0], [2.5]]})

        self.assertEqual(result["forecast"], [[2.5], [2.5], [2.5]])


if __name__ == "__main__":
    unittest.main()
