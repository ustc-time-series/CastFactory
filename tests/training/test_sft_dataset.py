import tempfile
import unittest
from pathlib import Path

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

    def test_sft_dataset_default_instruction_requires_json(self):
        from castfactory.representation import StatisticsRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        dataset = SFTDataset(
            samples=[self.make_sample()],
            representation=StatisticsRepresentation(features=["mean"]),
        )

        self.assertIn('Return JSON: {"forecast": [v1, ..., v2]}', dataset[0]["input"])

    def test_sft_dataset_includes_future_known_window(self):
        from castfactory.data.records import ForecastSample, TSRecord
        from castfactory.representation import StatisticsRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        observed = TSRecord(
            values=np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]]),
            timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
            channel_names=["load", "hour_of_day"],
            target_channels=["load"],
            covariate_channels=["hour_of_day"],
            static_context={},
            metadata={},
        )
        future_known = TSRecord(
            values=np.array([[13.0], [14.0]]),
            timestamps=pd.date_range("2022-01-01 03:00", periods=2, freq="h"),
            channel_names=["hour_of_day"],
            target_channels=[],
            covariate_channels=["hour_of_day"],
            static_context={},
            metadata={},
        )
        future_unknown = TSRecord(
            values=np.array([[4.0], [5.0]]),
            timestamps=future_known.timestamps,
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=future_known,
            future_unknown_window=future_unknown,
            cutoff_time=observed.timestamps[-1],
            prediction_length=2,
            metadata={},
        )

        item = SFTDataset([sample], StatisticsRepresentation(features=["mean"]))[0]

        self.assertIn("Known future covariates", item["input"])
        self.assertIn('"hour_of_day": [13.0, 14.0]', item["input"])

    def test_sft_dataset_instruction_template_can_use_significant_digits(self):
        from castfactory.representation import MarkdownTableRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        dataset = SFTDataset(
            samples=[self.make_sample()],
            representation=MarkdownTableRepresentation(significant_digits=2),
            instruction_template="Round values to {significant_digits} decimal places.",
        )

        self.assertIn("Round values to 2 decimal places.", dataset[0]["input"])

    def test_sft_dataset_template_can_embed_data_lookback_placeholders(self):
        from castfactory.representation import MarkdownTableRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        sample = self.make_sample()
        sample.observed_window.static_context["dataset_name"] = "ToySet"
        sample.observed_window.static_context["attr_meaning"] = "load"
        dataset = SFTDataset(
            samples=[sample],
            representation=MarkdownTableRepresentation(significant_digits=2),
            instruction_template=(
                "Dataset {dataset_name}; attr {attr_meaning}; look_back {look_back}; "
                "pred_window {pred_window}\n{data_lookback}"
            ),
        )

        rendered = dataset[0]["input"]

        self.assertIn("Dataset ToySet; attr load; look_back 3; pred_window 2", rendered)
        self.assertIn("| timestamp | load |", rendered)
        self.assertEqual(rendered.count("| timestamp | load |"), 1)

    def test_sft_dataset_can_load_instruction_template_from_txt_file(self):
        from castfactory.representation import MarkdownTableRepresentation
        from castfactory.training.sft_dataset import SFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            template_path = Path(tmp) / "prompt.txt"
            template_path.write_text("Forecast {pred_window} from {dataset_name}\n{data_lookback}")
            sample = self.make_sample()
            sample.observed_window.static_context["dataset_name"] = "ToySet"
            dataset = SFTDataset(
                samples=[sample],
                representation=MarkdownTableRepresentation(significant_digits=2),
                instruction_template=str(template_path),
            )

            rendered = dataset[0]["input"]

        self.assertIn("Forecast 2 from ToySet", rendered)
        self.assertIn("| timestamp | load |", rendered)


if __name__ == "__main__":
    unittest.main()
