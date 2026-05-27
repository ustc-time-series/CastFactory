import unittest

import numpy as np


class ForecastParserTests(unittest.TestCase):
    def test_json_parser_extracts_forecast_array(self):
        from castfactory.parsers import JSONForecastParser, ParseContext

        parser = JSONForecastParser()
        result = parser.parse(
            '{"forecast": [1.0, 2.5, 3.0]}',
            ParseContext(
                prediction_length=3,
                num_channels=1,
                output_schema="forecast_json_v1",
                channel_names=["load"],
            ),
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.point_forecast, np.array([[1.0], [2.5], [3.0]]))
        self.assertFalse(result.fallback_used)

    def test_array_parser_extracts_numbers_from_text(self):
        from castfactory.parsers import ArrayForecastParser, ParseContext

        parser = ArrayForecastParser()
        result = parser.parse(
            "Forecast: [4, 5.5, -6]",
            ParseContext(
                prediction_length=3,
                num_channels=1,
                output_schema="array",
                channel_names=["load"],
            ),
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.point_forecast, np.array([[4.0], [5.5], [-6.0]]))

    def test_think_answer_parser_extracts_numbers_from_answer_block(self):
        from castfactory.parsers import ParseContext, ThinkAnswerForecastParser

        parser = ThinkAnswerForecastParser()
        result = parser.parse(
            "<think>\nuse recent trend\n</think>\n"
            "<answer>\n```\n1.0\n2.5\n3.0\n```\n</answer>",
            ParseContext(
                prediction_length=3,
                num_channels=1,
                output_schema="think_answer_array",
                channel_names=["load"],
            ),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.fallback_used)
        np.testing.assert_allclose(result.point_forecast, np.array([[1.0], [2.5], [3.0]]))

    def test_timestamp_value_parser_ignores_timestamp_numbers(self):
        from castfactory.parsers import ParseContext, TimestampValueForecastParser

        parser = TimestampValueForecastParser()
        result = parser.parse(
            "<think>\nreason\n</think>\n"
            "<answer>\n"
            "2022-01-01 02:00:00 3.500\n"
            "2022-01-01 03:00:00 4.500\n"
            "</answer>",
            ParseContext(
                prediction_length=2,
                num_channels=1,
                output_schema="timestamp_value",
                channel_names=["OT"],
            ),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.fallback_used)
        np.testing.assert_allclose(result.point_forecast, np.array([[3.5], [4.5]]))

    def test_think_answer_parser_falls_back_when_answer_block_is_missing(self):
        from castfactory.parsers import ParseContext, ThinkAnswerForecastParser

        context = ParseContext(
            prediction_length=2,
            num_channels=1,
            output_schema="think_answer_array",
            channel_names=["load"],
            observed_values=np.array([[10.0], [12.0]]),
        )

        result = ThinkAnswerForecastParser().parse("<think>missing answer</think>", context)

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_strategy, "last_value")
        np.testing.assert_allclose(result.point_forecast, np.array([[12.0], [12.0]]))

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
