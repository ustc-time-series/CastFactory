import unittest

import numpy as np
import pandas as pd


class SFTDatasetTests(unittest.TestCase):
    def make_sample(self):
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0], [2.0], [3.0]]),
            timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={"domain": "energy"},
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

    def test_sft_dataset_formats_instruction_and_json_output(self):
        from castfactory.representation import StatisticsRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        dataset = SFTDataset(
            samples=[self.make_sample()],
            representation=StatisticsRepresentation(features=["mean"]),
            instruction_template="Predict the next {prediction_length} steps.",
        )

        item = dataset[0]

        self.assertIn("Predict the next 2 steps.", item["input"])
        self.assertIn("mean=2.0000", item["input"])
        self.assertEqual(item["output"], '{"forecast": [4.0, 5.0]}')


if __name__ == "__main__":
    unittest.main()
