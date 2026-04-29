from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


class RunStore:
    def __init__(self, root: str | Path = "runs", run_id: str = "default"):
        self.root = Path(root)
        self.run_id = run_id
        self.path = self.root / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload: Mapping) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_jsonl(self, relative_path: str, rows: Iterable[Mapping]) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        return path

    def write_leaderboard(self, metrics: Mapping[str, float]) -> Path:
        lines = ["# Leaderboard", "", "| metric | value |", "|---|---:|"]
        for name in sorted(metrics):
            lines.append(f"| {name} | {metrics[name]} |")
        return self._write_text("leaderboard.md", "\n".join(lines) + "\n")

    def write_report(self, title: str, metrics: Mapping[str, float]) -> Path:
        lines = [f"# {title}", "", "## Metrics", ""]
        for name in sorted(metrics):
            lines.append(f"- `{name}`: {metrics[name]}")
        return self._write_text("report.md", "\n".join(lines) + "\n")

    def _write_text(self, relative_path: str, text: str) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
