import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class CLITests(unittest.TestCase):
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
