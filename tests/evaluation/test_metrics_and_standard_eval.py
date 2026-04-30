import unittest

import math
import numpy as np
import pandas as pd


class MetricsAndStandardEvalTests(unittest.TestCase):
    def test_point_metrics_compute_expected_values(self):
        from castfactory.evaluation.metrics import mae, mape, mase, mse, smape

        target = np.array([[2.0], [4.0], [6.0]])
        pred = np.array([[1.0], [5.0], [7.0]])
        insample = np.array([[1.0], [2.0], [3.0], [4.0]])

        self.assertAlmostEqual(mae(pred, target), 1.0)
        self.assertAlmostEqual(mse(pred, target), 1.0)
        self.assertAlmostEqual(mape(pred, target), 0.3055555556)
        self.assertAlmostEqual(smape(pred, target), 0.3475783476)
        self.assertAlmostEqual(mase(pred, target, insample), 1.0)

    def test_mape_handles_zero_targets_without_nan(self):
        from castfactory.evaluation.metrics import mape

        self.assertEqual(mape(np.zeros((2, 1)), np.zeros((2, 1))), 0.0)
        self.assertTrue(math.isinf(mape(np.ones((2, 1)), np.zeros((2, 1)))))
        self.assertAlmostEqual(mape(np.array([[999.0], [2.0]]), np.array([[0.0], [4.0]])), 0.5)

    def test_llm_output_metrics_compute_rates(self):
        from castfactory.evaluation.metrics import format_valid_rate, parse_success_rate

        rows = [
            {"parse_success": True, "fallback_used": False},
            {"parse_success": False, "fallback_used": True},
            {"parse_success": True, "fallback_used": True},
        ]

        self.assertAlmostEqual(parse_success_rate(rows), 2 / 3)
        self.assertAlmostEqual(format_valid_rate(rows), 1 / 3)

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

    def test_standard_evaluator_rejects_empty_sample_sets(self):
        from castfactory.evaluation.protocols import StandardEvaluator

        with self.assertRaisesRegex(ValueError, "at least one evaluation sample"):
            StandardEvaluator(metrics=["mae"]).evaluate([], lambda samples: [])

    def test_standard_evaluator_rejects_wrong_forecast_shape(self):
        from castfactory.data.records import ForecastSample, TSRecord
        from castfactory.evaluation.protocols import StandardEvaluator

        observed = TSRecord(
            values=np.array([[1.0], [2.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
        )
        future = TSRecord(
            values=np.array([[3.0], [4.0]]),
            timestamps=pd.date_range("2022-01-01 02:00", periods=2, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=2,
        )

        with self.assertRaisesRegex(ValueError, "forecast shape"):
            StandardEvaluator(metrics=["mae"]).evaluate([sample], lambda samples: [np.array([1.0])])

    def test_rolling_evaluator_averages_metrics_per_origin(self):
        from castfactory.data.records import ForecastSample, TSRecord
        from castfactory.evaluation.protocols import RollingEvaluator, StandardEvaluator

        observed = TSRecord(
            values=np.array([[0.0], [0.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
        )
        first_future = TSRecord(
            values=np.array([[10.0]]),
            timestamps=pd.date_range("2022-01-01 02:00", periods=1, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
        )
        second_future = TSRecord(
            values=np.array([[0.0], [0.0], [0.0]]),
            timestamps=pd.date_range("2022-01-01 03:00", periods=3, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
        )
        samples = [
            ForecastSample(observed, None, first_future, observed.timestamps[-1], 1),
            ForecastSample(observed, None, second_future, observed.timestamps[-1], 3),
        ]

        def predictor(batch):
            return [np.zeros_like(sample.future_unknown_window.values) for sample in batch]

        standard = StandardEvaluator(metrics=["mae"]).evaluate(samples, predictor)
        rolling = RollingEvaluator(metrics=["mae"]).evaluate(samples, predictor)

        self.assertEqual(standard.metrics["mae"], 2.5)
        self.assertEqual(rolling.metrics["mae"], 5.0)
        self.assertEqual(rolling.metadata["protocol"], "rolling")

    def test_zero_shot_evaluator_marks_no_training_required(self):
        from castfactory.evaluation.protocols import ZeroShotEvaluator

        self.assertEqual(ZeroShotEvaluator(metrics=["mae"]).protocol_name, "zero_shot")
        self.assertFalse(ZeroShotEvaluator(metrics=["mae"]).training_required)


if __name__ == "__main__":
    unittest.main()
