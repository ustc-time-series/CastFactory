from __future__ import annotations


class VLLMEngineAdapter:
    def __init__(self, model_name: str, **kwargs):
        if not model_name:
            raise ValueError("model_name is required for VLLMEngineAdapter")
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        self.engine = None

    def load(self):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("VLLMEngineAdapter requires the optional 'vllm' dependency") from exc
        self.engine = LLM(model=self.model_name, **self.kwargs)
        return self

    def generate(self, prompts, **sampling_kwargs):
        if self.engine is None:
            self.load()
        return self.engine.generate(prompts, **sampling_kwargs)
