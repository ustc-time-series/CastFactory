from __future__ import annotations

import re

import numpy as np

from castfactory.parsers.array_parser import NUMBER_PATTERN
from castfactory.parsers.base import ParseContext, ParseResult, ParserBase
from castfactory.parsers.think_answer_parser import inspect_reasoning_answer_format


TIMESTAMP_VALUE_PATTERN = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s+"
    r"([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


class TimestampValueForecastParser(ParserBase):
    """Parse Cast-R1 style timestamp/value forecast answers."""

    def parse(self, raw_text: str, context: ParseContext) -> ParseResult:
        try:
            format_info = inspect_reasoning_answer_format(raw_text)
            payload = (
                str(format_info["answer_payload"]).strip()
                if format_info["has_answer_block"]
                else str(raw_text).strip()
            )
            values = []
            for raw_line in payload.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                line = line.strip("|").strip()
                match = TIMESTAMP_VALUE_PATTERN.match(line)
                if match:
                    values.append(float(match.group(1)))
                    continue
                numbers = [float(number.group(0)) for number in NUMBER_PATTERN.finditer(line)]
                if numbers:
                    values.append(numbers[-1])
            forecast = self._shape(np.array(values, dtype=float), context)
            return ParseResult(success=True, point_forecast=forecast, fallback_used=False)
        except Exception as exc:
            return self._fallback(context, str(exc))
