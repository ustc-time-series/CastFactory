import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class RunStoreTests(unittest.TestCase):
    def test_run_store_writes_metrics_predictions_and_reports(self):
        from castfactory.trace import RunStore

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(root=tmp, run_id="unit_run")
            store.write_json("metrics.json", {"mae": 0.5, "parse_success_rate": 1.0})
            store.write_jsonl("predictions.jsonl", [{"sample_id": "s1", "step": 1, "pred": 4.0}])
            store.write_leaderboard({"mae": 0.5, "mse": 1.0})
            store.write_report(title="Unit Run", metrics={"mae": 0.5})

            run_dir = Path(tmp) / "unit_run"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            leaderboard = (run_dir / "leaderboard.md").read_text()
            report = (run_dir / "report.md").read_text()

        self.assertEqual(metrics["mae"], 0.5)
        self.assertIn("| mae | 0.5 |", leaderboard)
        self.assertIn("# Unit Run", report)

    def test_run_store_has_semantic_trace_helpers(self):
        from castfactory.trace import RunStore

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(root=tmp, run_id="unit_run")
            predictions_path = store.save_predictions([
                {"sample_id": "s1", "step": 1, "pred": 4.0, "target": 5.0}
            ])
            store.save_prompts([{"sample_id": "s1", "prompt": "predict"}])
            store.save_responses([{"sample_id": "s1", "response": "{}"}])
            store.save_parsed([{"sample_id": "s1", "parse_success": True}])
            store.append_error({"sample_id": "s1", "message": "parse failed"})

            run_dir = Path(tmp) / "unit_run"
            frame = pd.read_parquet(predictions_path)
            errors = (run_dir / "errors.jsonl").read_text()

        self.assertEqual(predictions_path.name, "predictions.parquet")
        self.assertEqual(frame.loc[0, "pred"], 4.0)
        self.assertIn("parse failed", errors)

    def test_run_store_writes_stage_metadata(self):
        from castfactory.trace import RunStore

        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(root=tmp, run_id="run")
            path = store.save_stage_metadata(
                {
                    "stage": "sft",
                    "input_checkpoint": "checkpoints/cpt",
                    "output_checkpoint": "checkpoints/sft",
                }
            )

        self.assertEqual(path.name, "stage_metadata.json")


if __name__ == "__main__":
    unittest.main()
