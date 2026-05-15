import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class CLITests(unittest.TestCase):
    def test_parse_dotlist_overrides_converts_scalars_and_structures(self):
        from castfactory.cli.run import parse_dotlist_overrides

        overrides = parse_dotlist_overrides(
            [
                "training.args.learning_rate=2.0e-4",
                "training.backend.bf16=true",
                "model.peft.target_modules=[q_proj, v_proj]",
                "trace.run_root=null",
            ]
        )

        self.assertEqual(overrides["training"]["args"]["learning_rate"], 2.0e-4)
        self.assertIs(overrides["training"]["backend"]["bf16"], True)
        self.assertEqual(overrides["model"]["peft"]["target_modules"], ["q_proj", "v_proj"])
        self.assertIsNone(overrides["trace"]["run_root"])

    def test_apply_overrides_allows_empty_mapping_to_clear_nested_section(self):
        from castfactory.cli.run import apply_overrides

        merged = apply_overrides(
            {"model": {"backbone": {"model_name": "base"}, "peft": {"method": "lora"}}},
            {"model": {"backbone": {}}},
        )

        self.assertEqual(merged["model"]["backbone"], {})
        self.assertEqual(merged["model"]["peft"], {"method": "lora"})

    def test_cli_load_mode_applies_dotlist_overrides_to_recipe(self):
        from castfactory.cli.run import main

        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "recipe.yaml"
            recipe_path.write_text("experiment:\n  name: original\n", encoding="utf-8")
            output = io.StringIO()

            with patch(
                "sys.argv",
                [
                    "castfactory",
                    str(recipe_path),
                    "--mode",
                    "load",
                    "experiment.name=overridden",
                ],
            ):
                with redirect_stdout(output):
                    main()

        self.assertIn("Loaded CastFactory experiment: overridden", output.getvalue())

    def test_cli_can_execute_fit_mode(self):
        from castfactory.cli.run import main

        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "recipe.yaml"
            recipe_path.write_text("experiment:\n  name: cli_fit\n", encoding="utf-8")
            output = io.StringIO()

            with patch("sys.argv", ["castfactory", str(recipe_path), "--mode", "fit"]):
                with redirect_stdout(output):
                    main()

        self.assertIn('"status": "skipped"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
