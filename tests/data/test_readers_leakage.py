import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class ReadersLeakageTests(unittest.TestCase):
    def test_csv_reader_loads_timestamped_multichannel_record(self):
        from castfactory.data.readers import CSVReader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.csv"
            path.write_text(
                "date,OT,HUFL\n"
                "2022-01-01 00:00,1.0,10.0\n"
                "2022-01-01 01:00,2.0,11.0\n"
                "2022-01-01 02:00,3.0,12.0\n"
            )

            record = CSVReader(
                path=path,
                timestamp_col="date",
                target_channels=["OT"],
                covariate_channels=["HUFL"],
                static_context={"domain": "energy"},
            ).read()

        self.assertEqual(record.values.shape, (3, 2))
        self.assertEqual(record.channel_names, ["OT", "HUFL"])
        self.assertEqual(record.static_context["domain"], "energy")

    def test_csv_reader_excludes_undeclared_covariates_from_record(self):
        from castfactory.data.readers import CSVReader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "series.csv"
            path.write_text(
                "date,OT,HUFL,HULL\n"
                "2022-01-01 00:00,1.0,10.0,20.0\n"
                "2022-01-01 01:00,2.0,11.0,21.0\n"
                "2022-01-01 02:00,3.0,12.0,22.0\n"
            )

            record = CSVReader(
                path=path,
                timestamp_col="date",
                target_channels=["OT"],
            ).read()

        self.assertEqual(record.values.shape, (3, 1))
        self.assertEqual(record.channel_names, ["OT"])
        self.assertEqual(record.target_channels, ["OT"])
        self.assertEqual(record.covariate_channels, [])

    def test_leakage_checker_flags_future_label_at_or_before_cutoff(self):
        from castfactory.data.leakage import LeakageChecker
        from castfactory.data.records import ForecastSample, TSRecord

        observed = TSRecord(
            values=np.array([[1.0], [2.0]]),
            timestamps=pd.date_range("2022-01-01", periods=2, freq="h"),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        leaked_future = TSRecord(
            values=np.array([[3.0]]),
            timestamps=pd.DatetimeIndex(["2022-01-01 01:00"]),
            channel_names=["load"],
            target_channels=["load"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=leaked_future,
            cutoff_time=observed.timestamps[-1],
            prediction_length=1,
            metadata={},
        )

        issues = LeakageChecker().check_samples([sample])

        self.assertEqual(len(issues), 1)
        self.assertIn("future_unknown_window", issues[0].message)


if __name__ == "__main__":
    unittest.main()
