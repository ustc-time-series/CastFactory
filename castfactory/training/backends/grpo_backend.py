from __future__ import annotations


class GRPOBackend:
    def run(self, prompts, rewards) -> dict:
        return {
            "status": "skipped",
            "reason": "GRPO update step is not configured",
            "num_prompts": len(prompts),
            "num_rewards": len(rewards),
        }
