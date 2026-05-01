from __future__ import annotations

from typing import Iterable, List, Mapping


class RLVRDataset:
    def __init__(self, rows: Iterable[Mapping]):
        self.rows = [dict(row) for row in rows]
        for row in self.rows:
            if "prompt" not in row:
                raise ValueError("RLVRDataset rows must contain a prompt")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return dict(self.rows[index])

    def prompts(self) -> List[str]:
        return [str(row["prompt"]) for row in self.rows]
