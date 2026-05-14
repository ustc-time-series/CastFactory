from __future__ import annotations

from typing import Iterable, List

from castfactory.data.records import ForecastSample


class CPTDataset:
    def __init__(self, samples: Iterable[ForecastSample]):
        self.samples: List[ForecastSample] = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "text": self._sample_to_text(sample),
            "metadata": dict(sample.metadata),
        }

    def _sample_to_text(self, sample: ForecastSample) -> str:
        record = sample.observed_window
        parts = []
        domain = record.static_context.get("domain") or sample.metadata.get("domain")
        unit = record.static_context.get("unit") or sample.metadata.get("unit")
        if domain:
            parts.append(f"<domain> {domain}")
        if unit:
            parts.append(f"<unit> {unit}")
        parts.append("<channels> " + ",".join(record.channel_names))
        for channel_index, channel_name in enumerate(record.channel_names):
            values = []
            for timestamp, row in zip(record.timestamps, record.values):
                values.append(f"{timestamp}={float(row[channel_index])}")
            parts.append(f"<channel> {channel_name} <values> " + " ".join(values))
        return "\n".join(parts)
