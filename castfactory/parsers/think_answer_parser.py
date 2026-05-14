from __future__ import annotations

import re

from castfactory.parsers.array_parser import ArrayForecastParser
from castfactory.parsers.base import ParseContext, ParseResult, ParserBase


THINK_BLOCK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"```(?:[\w+-]+)?\s*(.*?)\s*```", re.DOTALL)


def inspect_reasoning_answer_format(raw_text: str) -> dict:
    think_match = THINK_BLOCK_PATTERN.search(raw_text)
    answer_match = ANSWER_BLOCK_PATTERN.search(raw_text)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    fence_match = CODE_FENCE_PATTERN.search(answer_content)
    payload = fence_match.group(1).strip() if fence_match else answer_content
    return {
        "has_think_block": think_match is not None,
        "has_answer_block": answer_match is not None,
        "has_answer_code_fence": fence_match is not None,
        "answer_payload": payload,
    }


class ThinkAnswerForecastParser(ParserBase):
    def parse(self, raw_text: str, context: ParseContext) -> ParseResult:
        format_info = inspect_reasoning_answer_format(raw_text)
        if not format_info["has_answer_block"]:
            return self._fallback(context, "response missing <answer>...</answer> block")
        answer_payload = str(format_info["answer_payload"]).strip()
        if not answer_payload:
            return self._fallback(context, "response answer block is empty")
        result = ArrayForecastParser().parse(answer_payload, context)
        if result.success:
            return result
        return self._fallback(context, result.parse_error or "could not parse answer payload")
