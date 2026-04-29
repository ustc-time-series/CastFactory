from __future__ import annotations

from typing import Iterable, Mapping


def parse_success_rate(rows: Iterable[Mapping]) -> float:
    row_list = list(rows)
    if not row_list:
        return 0.0
    successes = sum(1 for row in row_list if bool(row.get("parse_success", False)))
    return successes / len(row_list)


def format_valid_rate(rows: Iterable[Mapping]) -> float:
    row_list = list(rows)
    if not row_list:
        return 0.0
    valid = sum(
        1
        for row in row_list
        if bool(row.get("parse_success", False)) and not bool(row.get("fallback_used", False))
    )
    return valid / len(row_list)
