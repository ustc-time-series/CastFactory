from __future__ import annotations

import json
from typing import Any

import numpy as np

from castfactory.parsers.array_parser import ArrayForecastParser
from castfactory.parsers.base import ParseContext, ParseResult, ParserBase


class JSONForecastParser(ParserBase):
    def parse(self, raw_text: str, context: ParseContext) -> ParseResult:
        try:
            payload = self._load_json(raw_text)
            if not isinstance(payload, dict) or "forecast" not in payload:
                raise ValueError("JSON forecast output must contain a 'forecast' field")
            forecast = self._shape(np.asarray(payload["forecast"], dtype=float), context)
            quantiles = self._parse_quantiles(payload, context)
            return ParseResult(
                success=True,
                point_forecast=forecast,
                quantile_forecast=quantiles,
                fallback_used=False,
            )
        except Exception as exc:
            number_result = ArrayForecastParser().parse(raw_text, context)
            if number_result.success:
                return number_result
            return self._fallback(context, str(exc))

    def _load_json(self, raw_text: str) -> Any:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start : end + 1])

    def _parse_quantiles(self, payload: dict, context: ParseContext) -> dict:
        quantiles = payload.get("confidence") or payload.get("quantiles") or {}
        if not isinstance(quantiles, dict):
            return {}
        parsed = {}
        for key, value in quantiles.items():
            if key == "type":
                continue
            try:
                parsed[key] = self._shape(np.asarray(value, dtype=float), context)
            except Exception:
                continue
        return parsed
