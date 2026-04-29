import unittest

import numpy as np
import pandas as pd


class RecordsSplitsWindowsTests(unittest.TestCase):
    def make_record(self):
        from castfactory.data.records import TSRecord

        return TSRecord(
            values=np.arange(12, dtype=float).reshape(12, 1),
            timestamps=pd.date_range("2022-01-01", periods=12, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={"domain": "energy"},
            metadata={},
        )

    def test_tsrecord_requires_2d_values_matching_timestamps(self):
        from castfactory.data.records import TSRecord

        with self.assertRaisesRegex(ValueError, "2D"):
            TSRecord(
                values=np.arange(3),
                timestamps=pd.date_range("2022-01-01", periods=3, freq="h"),
                channel_names=["load"],
                target_channels=["load"],
                covariate_channels=[],
                static_context={},
                metadata={},
            )

    def test_timestamp_splitter_splits_without_overlap(self):
        from castfactory.data.splits import TimestampSplitter

        record = self.make_record()
        splitter = TimestampSplitter(
            train_end="2022-01-01 03:00",
            val_end="2022-01-01 07:00",
            test_end="2022-01-01 11:00",
        )

        split = splitter.split(record)

        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.val), 4)
        self.assertEqual(len(split.test), 4)
        self.assertLess(split.train.timestamps[-1], split.val.timestamps[0])

    def test_window_builder_creates_visibility_separated_samples(self):
        from castfactory.data.windows import WindowBuilder

        record = self.make_record()
        builder = WindowBuilder(context_length=4, prediction_length=2, stride=2)

        samples = builder.build(record)

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].observed_window.values[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(samples[0].future_unknown_window.values[:, 0].tolist(), [4, 5])
        self.assertEqual(samples[0].cutoff_time, record.timestamps[3])

    def test_window_builder_splits_future_known_covariates(self):
        from castfactory.data.records import TSRecord
        from castfactory.data.windows import WindowBuilder

        record = TSRecord(
            values=np.column_stack([
                np.arange(8, dtype=float),
                np.arange(100, 108, dtype=float),
            ]),
            timestamps=pd.date_range("2022-01-01", periods=8, freq="h"),
            channel_names=["load", "hour_of_day"],
            target_channels=["load"],
            covariate_channels=["hour_of_day"],
            static_context={},
            metadata={},
        )

        sample = WindowBuilder(context_length=4, prediction_length=2, stride=2).build(record)[0]

        self.assertEqual(sample.future_unknown_window.channel_names, ["load"])
        self.assertEqual(sample.future_unknown_window.values[:, 0].tolist(), [4.0, 5.0])
        self.assertIsNotNone(sample.future_known_window)
        self.assertEqual(sample.future_known_window.channel_names, ["hour_of_day"])
        self.assertEqual(sample.future_known_window.values[:, 0].tolist(), [104.0, 105.0])

    def test_train_only_normalizer_reuses_train_statistics(self):
        from castfactory.data.transforms import TrainOnlyStandardScaler

        train = self.make_record()
        test = self.make_record()
        scaler = TrainOnlyStandardScaler().fit(train)

        transformed = scaler.transform(test)

        self.assertAlmostEqual(float(transformed.values.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(transformed.values.std()), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
