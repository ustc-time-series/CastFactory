from __future__ import annotations


class PEFTAdapter:
    def __init__(self, method: str = "lora", **kwargs):
        self.method = method
        self.kwargs = dict(kwargs)

    def apply(self, model):
        if self.method != "lora":
            raise ValueError(f"Unsupported PEFT method: {self.method}")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("PEFTAdapter requires the optional 'peft' dependency") from exc
        config = LoraConfig(**self.kwargs)
        return get_peft_model(model, config)
