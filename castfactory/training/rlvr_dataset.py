from __future__ import annotations

from typing import Iterable, List, Mapping


class RLVRDataset:
    REQUIRED_FIELDS = (
        "prompt",
        "label",
        "prediction_length",
        "channel_names",
        "observed_values",
        "cutoff_time",
        "sample_id",
    )

    def __init__(self, rows: Iterable[Mapping]):
        self.rows = [dict(row) for row in rows]
        for row in self.rows:
            missing = [name for name in self.REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(f"RLVRDataset rows missing required fields: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return dict(self.rows[index])

    def prompts(self) -> List[str]:
        return [str(row["prompt"]) for row in self.rows]

    def rows_for_export(self) -> List[dict]:
        return [dict(row) for row in self.rows]
