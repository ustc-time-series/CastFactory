from __future__ import annotations

from castfactory.representation import ModelInput


class TextConcatBridge:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def build_prompt(self, model_input: ModelInput, instruction: str) -> str:
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(instruction)
        if model_input.text_prompt:
            parts.append(model_input.text_prompt)
        return "\n\n".join(parts)
