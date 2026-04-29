from __future__ import annotations

from castfactory.data.records import ForecastResult
from castfactory.parsers import ForecastParser, ParseContext


class TextGenerationHead:
    def __init__(self, parser: ForecastParser):
        self.parser = parser

    def generate(self, backbone, prompt: str, parse_context: ParseContext, **generation_kwargs) -> ForecastResult:
        raw_response = backbone.generate_text(prompt, **generation_kwargs)
        parsed = self.parser.parse(raw_response, parse_context)
        return ForecastResult(
            point_forecast=parsed.point_forecast,
            quantile_forecast=parsed.quantile_forecast,
            raw_response=raw_response,
            parse_success=parsed.success,
            fallback_used=parsed.fallback_used,
            metadata={
                "parse_error": parsed.parse_error,
                "fallback_strategy": parsed.fallback_strategy,
            },
        )
