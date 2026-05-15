from __future__ import annotations

from pathlib import Path


def load_instruction_template(template: str, *, base_dir: str | Path | None = None) -> str:
    candidate = Path(template)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / candidate
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return template


def build_instruction_format_kwargs(sample, model_input) -> dict:
    static_context = dict(sample.observed_window.static_context)
    target_channels = list(sample.future_unknown_window.channel_names)
    covariate_channels = list(sample.observed_window.covariate_channels)
    source_path = sample.observed_window.metadata.get("source_path")
    inferred_dataset_name = ""
    if source_path:
        inferred_dataset_name = Path(str(source_path)).stem
    format_kwargs = {
        "prediction_length": sample.prediction_length,
        "cutoff_time": sample.cutoff_time,
        "look_back": len(sample.observed_window),
        "pred_window": sample.prediction_length,
        "context_length": len(sample.observed_window),
        "dataset_name": static_context.get("dataset_name")
        or static_context.get("dataset")
        or static_context.get("domain")
        or inferred_dataset_name,
        "attr_meaning": static_context.get("attr_meaning")
        or static_context.get("target_description")
        or ",".join(target_channels),
        "target_channels": ",".join(target_channels),
        "covariate_channels": ",".join(covariate_channels),
        "data_lookback": model_input.text_prompt or "",
    }
    format_kwargs.update(static_context)
    format_kwargs.update(model_input.metadata or {})
    return format_kwargs


def template_uses_data_lookback(template: str) -> bool:
    return "{data_lookback}" in template
