import unittest

import numpy as np


class ModelSkeletonTests(unittest.TestCase):
    def test_text_concat_bridge_builds_prompt_from_model_input(self):
        from castfactory.models.bridges import TextConcatBridge
        from castfactory.representation import ModelInput

        bridge = TextConcatBridge(system_prompt="Forecasting task")

        prompt = bridge.build_prompt(
            ModelInput(text_prompt="statistics(channel=load, mean=2.0)"),
            instruction="Predict the next 2 steps.",
        )

        self.assertIn("Forecasting task", prompt)
        self.assertIn("Predict the next 2 steps.", prompt)
        self.assertIn("statistics(channel=load", prompt)

    def test_text_generation_head_parses_backbone_response(self):
        from castfactory.models.heads import TextGenerationHead
        from castfactory.parsers import JSONForecastParser, ParseContext

        class FakeBackbone:
            def generate_text(self, prompt, **kwargs):
                return '{"forecast": [1.0, 2.0]}'

        head = TextGenerationHead(parser=JSONForecastParser())

        result = head.generate(
            backbone=FakeBackbone(),
            prompt="Predict",
            parse_context=ParseContext(
                prediction_length=2,
                num_channels=1,
                output_schema="forecast_json_v1",
                channel_names=["load"],
            ),
        )

        self.assertTrue(result.parse_success)
        np.testing.assert_allclose(result.point_forecast, np.array([[1.0], [2.0]]))

    def test_hf_causal_lm_adapter_requires_model_name(self):
        from castfactory.models.backbones import HFCausalLMBackbone

        with self.assertRaisesRegex(ValueError, "model_name"):
            HFCausalLMBackbone(model_name="")


if __name__ == "__main__":
    unittest.main()
