import unittest

import numpy as np
import pandas as pd


class TextRepresentationTests(unittest.TestCase):
    def make_sample(self):
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0], [2.0], [3.0]]),
            timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={"domain": "energy", "freq": "1H"},
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
        return ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=2,
            metadata={},
        )

    def test_statistics_representation_describes_channel_statistics(self):
        from castfactory.representation import StatisticsRepresentation

        model_input = StatisticsRepresentation(features=["mean", "std"]).encode(self.make_sample())

        self.assertIn("load", model_input.text_prompt)
        self.assertIn("mean=2.0000", model_input.text_prompt)
        self.assertIn("std=", model_input.text_prompt)

    def test_hybrid_representation_combines_text_prompts(self):
        from castfactory.representation import ContextRepresentation, HybridRepresentation, StatisticsRepresentation

        representation = HybridRepresentation([
            ContextRepresentation(include_domain=True),
            StatisticsRepresentation(features=["mean"]),
        ])

        model_input = representation.encode(self.make_sample())

        self.assertIn("domain=energy", model_input.text_prompt)
        self.assertIn("mean=2.0000", model_input.text_prompt)


if __name__ == "__main__":
    unittest.main()
