from __future__ import annotations

import re

import numpy as np

from castfactory.parsers.base import ParseContext, ParseResult, ParserBase


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


class ArrayForecastParser(ParserBase):
    def parse(self, raw_text: str, context: ParseContext) -> ParseResult:
        try:
            numbers = [float(match.group(0)) for match in NUMBER_PATTERN.finditer(raw_text)]
            forecast = self._shape(np.array(numbers, dtype=float), context)
            return ParseResult(success=True, point_forecast=forecast, fallback_used=False)
        except Exception as exc:
            return self._fallback(context, str(exc))
