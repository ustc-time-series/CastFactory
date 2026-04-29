import json
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
