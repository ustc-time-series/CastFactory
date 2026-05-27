import asyncio
import importlib
import sys
import types
import unittest
from typing import Any
from unittest.mock import patch

import pandas as pd


class AgenticTimeSeriesToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_predict_time_series_uses_http_service_response(self):
        from castfactory.training.agentic.tools import predict_time_series_async

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "timestamps": ["2022-01-01 02:00:00", "2022-01-01 03:00:00"],
                    "values": [3.0, 4.0],
                }

        class Client:
            async def post(self, url, json):
                self.url = url
                self.payload = json
                return Response()

        client = Client()

        result = await predict_time_series_async(
            timestamps=[
                pd.Timestamp("2022-01-01 00:00:00"),
                pd.Timestamp("2022-01-01 01:00:00"),
            ],
            values=[1.0, 2.0],
            prediction_length=2,
            model_name="chronos2",
            service_url="http://model-service",
            http_client=client,
        )

        self.assertEqual(result["model_used"], "chronos2")
        self.assertEqual(result["timestamps"], ["2022-01-01 02:00:00", "2022-01-01 03:00:00"])
        self.assertEqual(result["values"], [3.0, 4.0])
        self.assertEqual(client.url, "http://model-service/predict")
        self.assertEqual(client.payload["prediction_length"], 2)

    async def test_predict_time_series_uses_arima_fallback_after_http_error(self):
        from castfactory.training.agentic import tools

        class Client:
            async def post(self, url, json):
                raise RuntimeError("service unavailable")

        def fake_arima(timestamps, values, prediction_length):
            return {
                "timestamps": ["2022-01-01 02:00:00", "2022-01-01 03:00:00"],
                "values": [2.5, 3.5],
                "model_used": "arima",
                "fallback_used": True,
            }

        with patch.object(tools, "_predict_with_arima_sync", fake_arima):
            result = await tools.predict_time_series_async(
                timestamps=[
                    pd.Timestamp("2022-01-01 00:00:00"),
                    pd.Timestamp("2022-01-01 01:00:00"),
                ],
                values=[1.0, 2.0],
                prediction_length=2,
                model_name="chronos2",
                service_url="http://model-service",
                http_client=Client(),
                local_fallback="arima_then_last_value",
            )

        self.assertEqual(result["model_used"], "arima")
        self.assertEqual(result["values"], [2.5, 3.5])
        self.assertEqual(result["fallback_used"], True)

    async def test_predict_time_series_uses_last_value_when_arima_fails(self):
        from castfactory.training.agentic import tools

        with patch.object(tools, "_predict_with_arima_sync", side_effect=RuntimeError("arima failed")):
            result = await tools.predict_time_series_async(
                timestamps=[
                    pd.Timestamp("2022-01-01 00:00:00"),
                    pd.Timestamp("2022-01-01 01:00:00"),
                ],
                values=[1.0, 2.0],
                prediction_length=2,
                model_name="arima",
                service_url=None,
                local_fallback="arima_then_last_value",
            )

        self.assertEqual(result["model_used"], "last_value")
        self.assertEqual(result["values"], [2.0, 2.0])
        self.assertEqual(result["timestamps"], ["2022-01-01 02:00:00", "2022-01-01 03:00:00"])

    def test_feature_extractors_return_sanitized_values(self):
        from castfactory.training.agentic.tools import (
            extract_basic_statistics,
            extract_data_quality,
            extract_event_summary,
            extract_forecast_residuals,
            extract_within_channel_dynamics,
        )

        values = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0]

        self.assertIn("median", extract_basic_statistics(values))
        self.assertIn("peak_count", extract_within_channel_dynamics(values))
        self.assertIn("residual_mean", extract_forecast_residuals(values))
        self.assertIn("quality_dropout_ratio", extract_data_quality(values))
        self.assertIn("event_segment_count", extract_event_summary(values))


class TimeSeriesAgentLoopImportTests(unittest.TestCase):
    def test_keeps_verl_agent_base_when_token_output_import_is_missing(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules(
            {
                "verl.workers": self._package("verl.workers"),
                "verl.workers.rollout": self._package("verl.workers.rollout"),
                "verl.workers.rollout.replica": None,
            }
        )

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                self.assertTrue(reloaded._VERL_AVAILABLE)
                self.assertIs(reloaded.TokenOutput, Any)
                self.assertTrue(hasattr(reloaded.TimeSeriesForecastAgentLoop, "process_multi_modal_info"))
        finally:
            importlib.reload(time_series_agent_loop)

    def test_keeps_verl_agent_base_when_tool_parser_import_is_missing(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules(
            {
                "verl.experimental.agent_loop.tool_parser": None,
            }
        )

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                self.assertTrue(reloaded._VERL_AVAILABLE)
                self.assertIsNone(reloaded.ToolParser)
                self.assertTrue(hasattr(reloaded.TimeSeriesForecastAgentLoop, "process_multi_modal_info"))
        finally:
            importlib.reload(time_series_agent_loop)

    def test_keeps_verl_agent_base_when_profiler_import_is_missing(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules(
            {
                "verl.utils": self._package("verl.utils"),
                "verl.utils.profiler": None,
            }
        )

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                self.assertTrue(reloaded._VERL_AVAILABLE)
                self.assertTrue(hasattr(reloaded.TimeSeriesForecastAgentLoop, "process_multi_modal_info"))
        finally:
            importlib.reload(time_series_agent_loop)

    def test_supports_verl_agent_base_with_process_vision_info_api(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules({})

        class ProcessVisionAgentLoopBase:
            async def process_vision_info(self, messages):
                return {}

        fake_modules["verl.experimental.agent_loop.agent_loop"].AgentLoopBase = ProcessVisionAgentLoopBase

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                self.assertTrue(reloaded._VERL_AVAILABLE)
                self.assertTrue(hasattr(reloaded.TimeSeriesForecastAgentLoop, "process_vision_info"))
                self.assertTrue(hasattr(reloaded.TimeSeriesForecastAgentLoop, "process_multi_modal_info"))
        finally:
            importlib.reload(time_series_agent_loop)

    def test_run_accepts_current_verl_process_vision_info_api(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules(
            {
                "verl.experimental.agent_loop.tool_parser": None,
            }
        )

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del token_ids, skip_special_tokens
                return "<answer>\n2022-01-01 01:00:00 2.000\n</answer>"

        class ServerManager:
            def __init__(self):
                self.generate_calls = []

            async def generate(
                self,
                request_id,
                *,
                prompt_ids,
                sampling_params,
                image_data=None,
                video_data=None,
            ):
                self.generate_calls.append(
                    {
                        "request_id": request_id,
                        "prompt_ids": prompt_ids,
                        "sampling_params": sampling_params,
                        "image_data": image_data,
                        "video_data": video_data,
                    }
                )
                return types.SimpleNamespace(token_ids=[3, 4], log_probs=[0.1, 0.2], extra_fields={})

        class CurrentVerlAgentLoopBase:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.rollout_config = types.SimpleNamespace(response_length=8)
                self.tokenizer = Tokenizer()
                self.server_manager = ServerManager()

            async def process_vision_info(self, messages):
                del messages
                return {"images": ["image"], "videos": ["video"]}

            async def apply_chat_template(
                self,
                messages,
                tools=None,
                images=None,
                videos=None,
                remove_system_prompt=False,
            ):
                del messages, tools, images, videos, remove_system_prompt
                return [1, 2]

        class FakeAgentLoopOutput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        agent_loop_module = fake_modules["verl.experimental.agent_loop.agent_loop"]
        agent_loop_module.AgentLoopBase = CurrentVerlAgentLoopBase
        agent_loop_module.AgentLoopOutput = FakeAgentLoopOutput

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                loop = reloaded.TimeSeriesForecastAgentLoop(max_steps=1)
                output = asyncio.run(
                    loop.run(
                        {},
                        raw_prompt=[{"role": "user", "content": "2022-01-01 00:00:00 1.000"}],
                        extra_info={"prediction_length": 1},
                    )
                )

                self.assertEqual(output.prompt_ids, [1, 2])
                self.assertEqual(output.response_ids, [3, 4])
                self.assertEqual(output.multi_modal_data, {"images": ["image"], "videos": ["video"]})
                self.assertEqual(loop.server_manager.generate_calls[0]["image_data"], ["image"])
                self.assertEqual(loop.server_manager.generate_calls[0]["video_data"], ["video"])
        finally:
            importlib.reload(time_series_agent_loop)

    def test_run_keeps_initial_prompt_ids_when_later_turn_prompt_exceeds_prompt_limit(self):
        from castfactory.training.agentic import time_series_agent_loop

        fake_modules = self._fake_verl_modules(
            {
                "verl.experimental.agent_loop.tool_parser": None,
            }
        )

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                if token_ids == [7]:
                    return "<answer>\n2022-01-01 01:00:00 2.000\n</answer>"
                return "call tool"

        class ServerManager:
            def __init__(self):
                self.generate_calls = 0

            async def generate(self, request_id, *, prompt_ids, sampling_params):
                del request_id, prompt_ids, sampling_params
                self.generate_calls += 1
                token_ids = [3] if self.generate_calls == 1 else [7]
                return types.SimpleNamespace(token_ids=token_ids, log_probs=None, extra_fields={})

        class CurrentVerlAgentLoopBase:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.rollout_config = types.SimpleNamespace(response_length=16)
                self.tokenizer = Tokenizer()
                self.server_manager = ServerManager()
                self.template_calls = 0

            async def process_vision_info(self, messages):
                del messages
                return {}

            async def apply_chat_template(
                self,
                messages,
                tools=None,
                images=None,
                videos=None,
                remove_system_prompt=False,
            ):
                del messages, tools, images, videos
                self.template_calls += 1
                if remove_system_prompt:
                    return [5, 6]
                if self.template_calls == 1:
                    return [1, 2]
                return list(range(5933))

        class Parser:
            def __init__(self):
                self.calls = 0

            async def extract_tool_calls(self, token_ids, tools):
                del token_ids, tools
                self.calls += 1
                if self.calls == 1:
                    return "", [types.SimpleNamespace(name="extract_basic_statistics", arguments="{}")]
                return "", []

        class FakeAgentLoopOutput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        agent_loop_module = fake_modules["verl.experimental.agent_loop.agent_loop"]
        agent_loop_module.AgentLoopBase = CurrentVerlAgentLoopBase
        agent_loop_module.AgentLoopOutput = FakeAgentLoopOutput

        try:
            with patch.dict(sys.modules, fake_modules, clear=False):
                reloaded = importlib.reload(time_series_agent_loop)
                loop = reloaded.TimeSeriesForecastAgentLoop(max_steps=2)
                loop.tool_parser = Parser()
                output = asyncio.run(
                    loop.run(
                        {},
                        raw_prompt=[{"role": "user", "content": "2022-01-01 00:00:00 1.000"}],
                        extra_info={"prediction_length": 1},
                    )
                )

                self.assertEqual(output.prompt_ids, [1, 2])
                self.assertEqual(output.response_ids, [3, 5, 6, 7])
                self.assertEqual(output.response_mask, [1, 0, 0, 1])
        finally:
            importlib.reload(time_series_agent_loop)

    def _fake_verl_modules(self, overrides):
        class FakeAgentLoopBase:
            async def process_multi_modal_info(self, messages):
                return {}

        class FakeAgentLoopOutput:
            pass

        class FakeToolParser:
            pass

        def fake_register(name):
            del name

            def decorator(cls):
                return cls

            return decorator

        def fake_simple_timer(name, metrics):
            del name, metrics

            class Timer:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Timer()

        agent_loop_module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
        agent_loop_module.AgentLoopBase = FakeAgentLoopBase
        agent_loop_module.AgentLoopOutput = FakeAgentLoopOutput
        agent_loop_module.register = fake_register

        tool_parser_module = types.ModuleType("verl.experimental.agent_loop.tool_parser")
        tool_parser_module.ToolParser = FakeToolParser

        profiler_module = types.ModuleType("verl.utils.profiler")
        profiler_module.simple_timer = fake_simple_timer

        fake_modules = {
            "verl": self._package("verl"),
            "verl.experimental": self._package("verl.experimental"),
            "verl.experimental.agent_loop": self._package("verl.experimental.agent_loop"),
            "verl.experimental.agent_loop.agent_loop": agent_loop_module,
            "verl.experimental.agent_loop.tool_parser": tool_parser_module,
            "verl.utils": self._package("verl.utils"),
            "verl.utils.profiler": profiler_module,
        }
        fake_modules.update(overrides)
        return fake_modules

    def _package(self, name):
        module = types.ModuleType(name)
        module.__path__ = []
        return module


if __name__ == "__main__":
    unittest.main()
