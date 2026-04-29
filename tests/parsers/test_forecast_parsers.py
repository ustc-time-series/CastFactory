import unittest

import numpy as np


class ForecastParserTests(unittest.TestCase):
    def test_json_parser_extracts_forecast_array(self):
        from castfactory.parsers import JSONForecastParser, ParseContext

        parser = JSONForecastParser()
        result = parser.parse(
            '{"forecast": [1.0, 2.5, 3.0]}',
            ParseContext(prediction_length=3, num_channels=1, output_schema="forecast_json_v1", channel_names=["load"]),
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.point_forecast, np.array([[1.0], [2.5], [3.0]]))
        self.assertFalse(result.fallback_used)

    def test_array_parser_extracts_numbers_from_text(self):
        from castfactory.parsers import ArrayForecastParser, ParseContext

        parser = ArrayForecastParser()
        result = parser.parse(
            "Forecast: [4, 5.5, -6]",
            ParseContext(prediction_length=3, num_channels=1, output_schema="array", channel_names=["load"]),
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.point_forecast, np.array([[4.0], [5.5], [-6.0]]))

    def test_parser_fallback_uses_last_observed_value(self):
        from castfactory.parsers import JSONForecastParser, ParseContext

        context = ParseContext(
            prediction_length=2,
            num_channels=1,
            output_schema="forecast_json_v1",
            channel_names=["load"],
            observed_values=np.array([[10.0], [12.0]]),
        )

        result = JSONForecastParser().parse("no valid forecast here", context)

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_strategy, "last_value")
        np.testing.assert_allclose(result.point_forecast, np.array([[12.0], [12.0]]))


if __name__ == "__main__":
    unittest.main()
