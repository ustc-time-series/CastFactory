import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
