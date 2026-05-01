from __future__ import annotations


class HFCausalLMBackbone:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        device_map=None,
        torch_dtype=None,
        **generation_defaults,
    ):
        if not model_name:
            raise ValueError("model_name is required for HFCausalLMBackbone")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.generation_defaults = dict(generation_defaults)
        self.model = None
        self.tokenizer = None

    def load(self, tokenizer_factory=None, model_factory=None):
        if tokenizer_factory is None or model_factory is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "HFCausalLMBackbone requires the optional 'transformers' dependency"
                ) from exc
            tokenizer_factory = tokenizer_factory or AutoTokenizer
            model_factory = model_factory or AutoModelForCausalLM
        tokenizer_kwargs = self._common_load_kwargs()
        model_kwargs = dict(tokenizer_kwargs)
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        try:
            self.tokenizer = tokenizer_factory.from_pretrained(
                self.model_name,
                **tokenizer_kwargs,
            )
            self.model = model_factory.from_pretrained(self.model_name, **model_kwargs)
        except OSError as exc:
            raise OSError(
                "Failed to load HuggingFace causal LM "
                f"'{self.model_name}'. If running offline, set local_files_only=True "
                "and ensure the model is already cached or provide cache_dir."
            ) from exc
        return self

    def _common_load_kwargs(self) -> dict:
        kwargs = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def generate_text(self, prompt: str, **kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        options = dict(self.generation_defaults)
        options.update(kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs)
        outputs = self.model.generate(**inputs, **options)
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        if hasattr(generated_tokens, "tolist"):
            generated_tokens = generated_tokens.tolist()
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _move_inputs_to_model_device(self, inputs: dict) -> dict:
        device = self._model_device()
        if device is None:
            return inputs
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    def _model_device(self):
        if not hasattr(self.model, "parameters"):
            return None
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None
