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

    def test_textual_summary_generates_readable_time_series_description(self):
        from castfactory.representation import TextualSummaryRepresentation

        model_input = TextualSummaryRepresentation().encode(self.make_sample())

        self.assertIn("The time series spans", model_input.text_prompt)
        self.assertIn('Channel "load"', model_input.text_prompt)
        self.assertIn("Recent trend: upward", model_input.text_prompt)
        self.assertIn("Last observed value: 3.0000", model_input.text_prompt)

    def test_hybrid_representation_preserves_embeddings(self):
        from castfactory.representation import HybridRepresentation, ModelInput

        class EmbeddingComponent:
            def __init__(self, value):
                self.value = value

            def encode(self, sample):
                return ModelInput(embeddings=np.array([[self.value, self.value + 1.0]]))

        model_input = HybridRepresentation([
            EmbeddingComponent(1.0),
            EmbeddingComponent(3.0),
        ]).encode(self.make_sample())

        np.testing.assert_allclose(model_input.embeddings, np.array([[1.0, 2.0], [3.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
