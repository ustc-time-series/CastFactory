from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np


@dataclass
class ParseContext:
    prediction_length: int
    num_channels: int
    output_schema: str
    channel_names: list[str]
    observed_values: Optional[np.ndarray] = None
    fallback_strategy: str = "last_value"


@dataclass
class ParseResult:
    success: bool
    point_forecast: Optional[np.ndarray]
    quantile_forecast: Optional[Dict[str, np.ndarray]] = None
    parse_error: Optional[str] = None
    fallback_used: bool = False
    fallback_strategy: Optional[str] = None


class ForecastParser(Protocol):
    def parse(self, raw_text: str, context: ParseContext) -> ParseResult:
        ...


class ParserBase:
    def _shape(self, values: np.ndarray, context: ParseContext) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        expected = context.prediction_length * context.num_channels
        flat = values.reshape(-1)
        if flat.size < expected:
            raise ValueError(
                f"expected at least {expected} forecast values, received {flat.size}"
            )
        return flat[:expected].reshape(context.prediction_length, context.num_channels)

    def _fallback(self, context: ParseContext, error: str) -> ParseResult:
        strategy = context.fallback_strategy
        observed = (
            None
            if context.observed_values is None
            else np.asarray(context.observed_values, dtype=float)
        )
        if strategy == "last_value" and observed is not None and observed.size:
            last = observed.reshape(-1, context.num_channels)[-1]
            forecast = np.tile(last, (context.prediction_length, 1))
        elif strategy == "mean" and observed is not None and observed.size:
            mean = observed.reshape(-1, context.num_channels).mean(axis=0)
            forecast = np.tile(mean, (context.prediction_length, 1))
        elif strategy == "zero":
            forecast = np.zeros((context.prediction_length, context.num_channels), dtype=float)
        elif observed is not None and observed.size:
            last = observed.reshape(-1, context.num_channels)[-1]
            forecast = np.tile(last, (context.prediction_length, 1))
            strategy = "last_value"
        else:
            forecast = np.zeros((context.prediction_length, context.num_channels), dtype=float)
            strategy = "zero"
        return ParseResult(
            success=False,
            point_forecast=forecast,
            parse_error=error,
            fallback_used=True,
            fallback_strategy=strategy,
        )
