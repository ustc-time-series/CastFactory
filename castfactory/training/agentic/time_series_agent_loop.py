from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from castfactory.training.agentic.tools import (
    CAST_R1_TOOL_SCHEMAS,
    TIMESTAMP_VALUE_LINE,
    extract_basic_statistics,
    extract_data_quality,
    extract_event_summary,
    extract_forecast_residuals,
    extract_within_channel_dynamics,
    parse_time_series_string,
    predict_time_series_async,
)

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    _VERL_AVAILABLE = True
    _VERL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when verl is unavailable.
    _VERL_AVAILABLE = False
    _VERL_IMPORT_ERROR = exc
    ToolParser = None

    class AgentLoopBase:  # type: ignore[no-redef]
        pass

    class AgentLoopOutput:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def register(name):  # type: ignore[no-redef]
        del name

        def decorator(cls):
            return cls

        return decorator

try:
    from verl.experimental.agent_loop.tool_parser import ToolParser
except Exception:  # pragma: no cover - optional across verl versions.
    ToolParser = None

try:
    from verl.utils.profiler import simple_timer
except Exception:  # pragma: no cover - optional across verl versions.
    @contextmanager
    def simple_timer(name, metrics):  # type: ignore[no-redef]
        del name, metrics
        yield

try:
    from verl.workers.rollout.replica import TokenOutput
except Exception:  # pragma: no cover - TokenOutput is only used for typing.
    TokenOutput = Any


FEATURE_TOOLS = {
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
}


@register("time_series_forecast_agent")
class TimeSeriesForecastAgentLoop(AgentLoopBase):
    """Native verl AgentLoop for Cast-R1 style time-series forecasting."""

    def __init__(
        self,
        *args,
        max_steps: int = 3,
        max_parallel_calls: int = 5,
        tool_parser_format: str = "hermes",
        model_service_url: str | None = "http://localhost:8994",
        prediction_models: list[str] | None = None,
        local_fallback: str = "arima_then_last_value",
        **kwargs,
    ):
        if _VERL_AVAILABLE:
            super().__init__(*args, **kwargs)
        self.max_steps = int(max_steps)
        self.max_parallel_calls = int(max_parallel_calls)
        self.tool_parser_format = tool_parser_format
        self.model_service_url = model_service_url
        self.prediction_models = prediction_models or ["chronos2", "arima", "patchtst", "itransformer"]
        self.local_fallback = local_fallback
        self.tool_schemas = CAST_R1_TOOL_SCHEMAS
        self.tool_parser = None
        if ToolParser is not None and hasattr(self, "tokenizer"):
            self.tool_parser = ToolParser.get_tool_parser(tool_parser_format, self.tokenizer)
        rollout_config = getattr(self, "rollout_config", None)
        self.response_length = int(getattr(rollout_config, "response_length", 8192))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if not _VERL_AVAILABLE:
            detail = f" ({_VERL_IMPORT_ERROR!r})" if _VERL_IMPORT_ERROR is not None else ""
            raise RuntimeError(
                f"verl AgentLoopBase is unavailable{detail}; install a compatible verl runtime before "
                "running TimeSeriesForecastAgentLoop."
            ) from _VERL_IMPORT_ERROR
        messages = list(kwargs["raw_prompt"])
        extra_info = dict(kwargs.get("extra_info", {}) or {})
        observed_text = _raw_user_content(messages)
        timestamps, values = _extract_observed_series(observed_text, extra_info)
        prediction_length = int(extra_info.get("prediction_length", 1))

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids: list[int] = []
        initial_prompt_ids: list[int] = []
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        metrics: dict[str, Any] = {}
        feature_analysis_called = False
        prediction_called = False
        workflow_valid = False
        workflow_violation = "max_steps_exhausted"
        prediction_results: list[dict[str, Any]] = []
        request_id = uuid4().hex

        for _ in range(self.max_steps):
            prompt_ids = await self._apply_chat_template(
                messages,
                tools=self.tool_schemas,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            if not initial_prompt_ids:
                initial_prompt_ids = list(prompt_ids)
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self._generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            turn_ids = list(output.token_ids)
            response_ids.extend(turn_ids)
            response_mask.extend([1] * len(turn_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs)

            assistant_text = self.tokenizer.decode(turn_ids, skip_special_tokens=False)
            if "<answer>" in assistant_text.lower():
                workflow_valid, workflow_violation = _validate_workflow(
                    assistant_text=assistant_text,
                    observed_timestamps=timestamps,
                    feature_analysis_called=feature_analysis_called,
                    prediction_called=prediction_called,
                )
                break

            tool_calls = await self._extract_tool_calls(turn_ids)
            if not tool_calls:
                feedback_ids = await self._append_tool_feedback(
                    messages,
                    "Call at least one analysis tool and one prediction tool before finalizing.",
                    images,
                    videos,
                    audios,
                    mm_processor_kwargs,
                )
                response_ids.extend(feedback_ids)
                response_mask.extend([0] * len(feedback_ids))
                if response_logprobs:
                    response_logprobs.extend([0.0] * len(feedback_ids))
                continue

            tool_messages = []
            tasks = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                tasks.append(
                    self._execute_tool_call(
                        tool_call,
                        timestamps=timestamps,
                        values=values,
                        prediction_length=prediction_length,
                    )
                )
            for name, payload in await asyncio.gather(*tasks):
                if name in FEATURE_TOOLS:
                    feature_analysis_called = True
                if name == "predict_time_series":
                    prediction_called = True
                prediction_results.append(payload)
                tool_messages.append({"role": "tool", "content": json.dumps(payload, sort_keys=True)})
            messages.extend(tool_messages)
            tool_ids = await self._apply_chat_template(tool_messages, remove_system_prompt=True)
            response_ids.extend(tool_ids)
            response_mask.extend([0] * len(tool_ids))
            if response_logprobs:
                response_logprobs.extend([0.0] * len(tool_ids))
            if len(response_ids) >= self.response_length:
                break

        extra_fields = dict(getattr(output, "extra_fields", {}) or {}) if "output" in locals() else {}
        extra_fields.update(
            {
                "feature_analysis_called": bool(feature_analysis_called),
                "prediction_called": bool(prediction_called),
                "workflow_valid": bool(workflow_valid),
                "workflow_violation": workflow_violation,
                "workflow_penalty": 0.0 if workflow_valid else -0.5,
                "prediction_results": prediction_results,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length] if response_logprobs else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=len(messages),
            metrics=metrics,
            extra_fields=extra_fields,
        )

    async def process_multi_modal_info(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        method = getattr(super(), "process_multi_modal_info", None)
        if method is None:
            method = getattr(super(), "process_vision_info", None)
        if method is None:
            return {}
        result = method(messages)
        if inspect.isawaitable(result):
            return await result
        return result

    def _get_mm_processor_kwargs(self, audios) -> dict[str, Any] | None:
        method = getattr(super(), "_get_mm_processor_kwargs", None)
        if method is None:
            return None
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(audios)
        params = signature.parameters
        accepts_positional = any(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in params.values()
        )
        if accepts_positional:
            return method(audios)
        return method()

    async def _apply_chat_template(self, messages: list[dict[str, Any]], **kwargs) -> list[int]:
        return await self.apply_chat_template(
            **_supported_kwargs(
                self.apply_chat_template,
                {
                    "messages": messages,
                    **kwargs,
                },
            )
        )

    async def _generate(self, **kwargs) -> TokenOutput:
        return await self.server_manager.generate(
            **_supported_kwargs(self.server_manager.generate, kwargs)
        )

    async def _extract_tool_calls(self, token_ids: list[int]) -> list[Any]:
        if self.tool_parser is None:
            return []
        tools = [schema["function"] for schema in self.tool_schemas]
        _, tool_calls = await self.tool_parser.extract_tool_calls(token_ids, tools)
        return list(tool_calls or [])

    async def _execute_tool_call(
        self,
        tool_call: Any,
        *,
        timestamps: list[str],
        values: list[float],
        prediction_length: int,
    ) -> tuple[str, dict[str, Any]]:
        name = str(getattr(tool_call, "name", ""))
        try:
            arguments = json.loads(getattr(tool_call, "arguments", "{}") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if name == "extract_basic_statistics":
            return name, extract_basic_statistics(timestamps, values)
        if name == "extract_within_channel_dynamics":
            return name, extract_within_channel_dynamics(timestamps, values)
        if name == "extract_forecast_residuals":
            return name, extract_forecast_residuals(timestamps, values)
        if name == "extract_data_quality":
            return name, extract_data_quality(timestamps, values)
        if name == "extract_event_summary":
            return name, extract_event_summary(timestamps, values)
        if name == "predict_time_series":
            model_name = str(arguments.get("model_name") or self.prediction_models[0])
            return name, await predict_time_series_async(
                timestamps,
                values,
                prediction_length,
                model_name=model_name,
                service_url=self.model_service_url,
                local_fallback=self.local_fallback,
            )
        return name or "unknown_tool", {"error": f"unknown tool: {name}"}

    async def _append_tool_feedback(
        self,
        messages: list[dict[str, Any]],
        content: str,
        images,
        videos,
        audios,
        mm_processor_kwargs,
    ) -> list[int]:
        feedback = [{"role": "tool", "content": content}]
        messages.extend(feedback)
        return await self._apply_chat_template(
            feedback,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            remove_system_prompt=True,
        )


def _raw_user_content(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )


def _extract_observed_series(text: str, extra_info: dict[str, Any]) -> tuple[list[str], list[float]]:
    try:
        return parse_time_series_string(text)
    except ValueError:
        timestamps = list(extra_info.get("observed_timestamps", []))
        values = extra_info.get("observed_values", [])
        flat_values = [float(row[0] if isinstance(row, list) else row) for row in values]
        return timestamps, flat_values


def _validate_workflow(
    *,
    assistant_text: str,
    observed_timestamps: list[str],
    feature_analysis_called: bool,
    prediction_called: bool,
) -> tuple[bool, str]:
    if not feature_analysis_called:
        return False, "missing_feature_analysis_tool"
    if not prediction_called:
        return False, "missing_prediction_tool"
    answer_timestamps = []
    for line in assistant_text.splitlines():
        match = TIMESTAMP_VALUE_LINE.match(line.strip())
        if match:
            answer_timestamps.append(match.group(1).replace("T", " "))
    observed = {timestamp.replace("T", " ") for timestamp in observed_timestamps}
    if answer_timestamps and any(timestamp in observed for timestamp in answer_timestamps):
        return False, "copied_observed_timestamp"
    return True, ""


def _supported_kwargs(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
