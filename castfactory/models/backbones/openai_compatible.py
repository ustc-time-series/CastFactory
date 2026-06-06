from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAICompatibleBackbone:
    """Minimal OpenAI-compatible HTTP client for vLLM evaluation."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:12000/v1",
        api_key: str = "EMPTY",
        endpoint: str = "chat",
        timeout: float = 60,
        max_retries: int = 2,
        retry_sleep: float = 0.25,
        **generation_defaults,
    ):
        if not model_name:
            raise ValueError("model_name is required for OpenAICompatibleBackbone")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.generation_defaults = dict(generation_defaults)

    def load(self):
        return self

    def generate_text(self, prompt: str, **kwargs) -> str:
        options = dict(self.generation_defaults)
        options.update(kwargs)
        payload = self._payload(prompt, options)
        response = self._post_json(self._endpoint_url(), payload)
        return self._extract_text(response)

    def _payload(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            **{key: value for key, value in options.items() if value is not None},
        }
        if self._endpoint_kind() == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        return payload

    def _endpoint_url(self) -> str:
        kind = self._endpoint_kind()
        if kind == "chat":
            return f"{self.base_url}/chat/completions"
        if kind == "completion":
            return f"{self.base_url}/completions"
        raise ValueError("endpoint must be one of: chat, completion")

    def _endpoint_kind(self) -> str:
        endpoint = str(self.endpoint).strip().lower().replace("_", "-")
        if endpoint in {"chat", "chat-completions", "chat/completions"}:
            return "chat"
        if endpoint in {"completion", "completions"}:
            return "completion"
        return endpoint

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=data, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"OpenAI-compatible generation request failed: {last_error}")

    def _extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not choices:
            raise ValueError("OpenAI-compatible response missing choices")
        first = choices[0]
        if self._endpoint_kind() == "chat":
            message = first.get("message") or {}
            content = message.get("content")
        else:
            content = first.get("text")
        if content is None:
            raise ValueError("OpenAI-compatible response missing generated text")
        return str(content)
