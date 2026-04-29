import unittest

import numpy as np
import pandas as pd


class MetricsAndStandardEvalTests(unittest.TestCase):
    def test_point_metrics_compute_expected_values(self):
        from castfactory.evaluation.metrics import mae, mase, mse, smape

        target = np.array([[2.0], [4.0], [6.0]])
        pred = np.array([[1.0], [5.0], [7.0]])
        insample = np.array([[1.0], [2.0], [3.0], [4.0]])

        self.assertAlmostEqual(mae(pred, target), 1.0)
        self.assertAlmostEqual(mse(pred, target), 1.0)
        self.assertAlmostEqual(smape(pred, target), 0.3475783476)
        self.assertAlmostEqual(mase(pred, target, insample), 1.0)

    def test_standard_evaluator_collects_predictions_and_metrics(self):
        from castfactory.data.records import ForecastSample, TSRecord
        from castfactory.evaluation.protocols import StandardEvaluator

        observed = TSRecord(
            values=np.array([[1.0], [2.0], [3.0]]),
            timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        future = TSRecord(
            values=np.array([[4.0], [5.0]]),
            timestamps=pd.date_range("2022-01-01 03:00", periods=2, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
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
            metadata={"sample_id": "s1"},
        )

        def predictor(batch):
            return [np.array([[4.0], [6.0]]) for _ in batch]

        result = StandardEvaluator(metrics=["mae", "mse"]).evaluate([sample], predictor)

        self.assertEqual(result.metrics["mae"], 0.5)
        self.assertEqual(result.metrics["mse"], 0.5)
        self.assertEqual(len(result.predictions), 2)
        self.assertEqual(result.predictions[0]["sample_id"], "s1")
        self.assertEqual(result.predictions[1]["step"], 2)


if __name__ == "__main__":
    unittest.main()
