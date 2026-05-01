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

    def test_hf_causal_lm_decodes_only_new_tokens(self):
        from castfactory.models.backbones import HFCausalLMBackbone

        class FakeTokenizer:
            def __init__(self):
                self.decoded_tokens = None

            def __call__(self, prompt, return_tensors):
                return {"input_ids": np.array([[10, 11, 12]])}

            def decode(self, tokens, skip_special_tokens):
                self.decoded_tokens = list(tokens)
                return "forecast only"

        class FakeModel:
            def generate(self, **kwargs):
                return np.array([[10, 11, 12, 21, 22]])

        backbone = HFCausalLMBackbone(model_name="fake")
        backbone.tokenizer = FakeTokenizer()
        backbone.model = FakeModel()

        text = backbone.generate_text("prompt")

        self.assertEqual(text, "forecast only")
        self.assertEqual(backbone.tokenizer.decoded_tokens, [21, 22])

    def test_hf_causal_lm_passes_loading_options(self):
        from castfactory.models.backbones import HFCausalLMBackbone

        class FakeTokenizerFactory:
            kwargs = None

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                cls.kwargs = {"model_name": model_name, **kwargs}
                return object()

        class FakeModelFactory:
            kwargs = None

            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                cls.kwargs = {"model_name": model_name, **kwargs}
                return object()

        backbone = HFCausalLMBackbone(
            model_name="local-model",
            cache_dir="/tmp/hf-cache",
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        backbone.load(
            tokenizer_factory=FakeTokenizerFactory,
            model_factory=FakeModelFactory,
        )

        self.assertEqual(FakeTokenizerFactory.kwargs["model_name"], "local-model")
        self.assertEqual(FakeTokenizerFactory.kwargs["cache_dir"], "/tmp/hf-cache")
        self.assertTrue(FakeTokenizerFactory.kwargs["local_files_only"])
        self.assertTrue(FakeTokenizerFactory.kwargs["trust_remote_code"])
        self.assertEqual(FakeModelFactory.kwargs["device_map"], "auto")
        self.assertEqual(FakeModelFactory.kwargs["torch_dtype"], "auto")

    def test_hf_causal_lm_moves_inputs_to_model_device(self):
        from castfactory.models.backbones import HFCausalLMBackbone

        class FakeTensor:
            def __init__(self, values):
                self.values = values
                self.device = None

            @property
            def shape(self):
                return (1, len(self.values))

            def to(self, device):
                self.device = device
                return self

            def __getitem__(self, index):
                if isinstance(index, slice):
                    return self.values[index]
                return self.values[index]

        class FakeParameter:
            device = "cuda:0"

        class FakeTokenizer:
            def __init__(self):
                self.input_ids = FakeTensor([10, 11])

            def __call__(self, prompt, return_tensors):
                return {"input_ids": self.input_ids}

            def decode(self, tokens, skip_special_tokens):
                return "ok"

        class FakeModel:
            def __init__(self):
                self.received = None

            def parameters(self):
                return iter([FakeParameter()])

            def generate(self, **kwargs):
                self.received = kwargs
                return [FakeTensor([10, 11, 12])]

        backbone = HFCausalLMBackbone(model_name="fake")
        backbone.tokenizer = FakeTokenizer()
        backbone.model = FakeModel()

        self.assertEqual(backbone.generate_text("prompt"), "ok")
        self.assertEqual(backbone.tokenizer.input_ids.device, "cuda:0")

    def test_projector_bridge_projects_embeddings(self):
        from castfactory.models.bridges import ProjectorBridge
        from castfactory.representation import ModelInput

        bridge = ProjectorBridge(weights=np.array([[1.0, 0.0], [0.0, 2.0]]))
        output = bridge.project(ModelInput(embeddings=np.array([[3.0, 4.0]])))

        np.testing.assert_allclose(output.embeddings, np.array([[3.0, 8.0]]))


if __name__ == "__main__":
    unittest.main()
