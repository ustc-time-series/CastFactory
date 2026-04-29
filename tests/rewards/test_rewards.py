import unittest

import numpy as np


class RewardTests(unittest.TestCase):
    def test_accuracy_reward_returns_negative_metric_value(self):
        from castfactory.rewards import AccuracyReward

        reward = AccuracyReward(metric="mae")
        result = reward.compute(pred=np.array([[1.0], [3.0]]), target=np.array([[2.0], [3.0]]))

        self.assertEqual(result.name, "accuracy")
        self.assertEqual(result.value, -0.5)
        self.assertEqual(result.details["metric"], "mae")

    def test_format_reward_checks_parse_success_and_shape(self):
        from castfactory.parsers import ParseResult
        from castfactory.rewards import FormatReward

        parsed = ParseResult(
            success=True,
            point_forecast=np.array([[1.0], [2.0]]),
            fallback_used=False,
        )

        result = FormatReward(prediction_length=2, num_channels=1).compute(parsed)

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.details["shape_valid"], True)

    def test_composite_reward_weighted_sum(self):
        from castfactory.rewards import CompositeReward, RewardResult

        reward = CompositeReward(weights={"format": 0.25, "accuracy": 0.75})
        result = reward.combine([
            RewardResult(name="format", value=1.0, details={}),
            RewardResult(name="accuracy", value=-0.5, details={}),
        ])

        self.assertEqual(result.name, "composite")
        self.assertAlmostEqual(result.value, -0.125)
        self.assertEqual(result.details["format"], 1.0)

    def test_composite_reward_can_clip_to_unit_range(self):
        from castfactory.rewards import CompositeReward, RewardResult

        reward = CompositeReward(weights={"accuracy": 1.0}, normalize=True)
        result = reward.combine([RewardResult(name="accuracy", value=-10.0, details={})])

        self.assertEqual(result.value, -1.0)

    def test_calibration_reward_scores_interval_coverage(self):
        from castfactory.rewards import CalibrationReward

        reward = CalibrationReward()
        result = reward.compute(
            quantile_forecast={
                "q10": np.array([[0.0], [2.0]]),
                "q90": np.array([[2.0], [4.0]]),
            },
            target=np.array([[1.0], [5.0]]),
        )

        self.assertEqual(result.name, "calibration")
        self.assertEqual(result.details["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
