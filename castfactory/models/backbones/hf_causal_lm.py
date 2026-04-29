from __future__ import annotations


class HFCausalLMBackbone:
    def __init__(self, model_name: str, **generation_defaults):
        if not model_name:
            raise ValueError("model_name is required for HFCausalLMBackbone")
        self.model_name = model_name
        self.generation_defaults = dict(generation_defaults)
        self.model = None
        self.tokenizer = None

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFCausalLMBackbone requires the optional 'transformers' dependency"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return self

    def generate_text(self, prompt: str, **kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        options = dict(self.generation_defaults)
        options.update(kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
