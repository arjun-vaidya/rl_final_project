"""Flat (non-hierarchical) CoT+tool agent.

Condition #2 baseline in docs/05_evaluation.md.
Emits reasoning with interleaved <code>...</code> blocks; env executes each as it appears.
Final answer parsed from <answer>X</answer>.
"""
from dataclasses import dataclass


@dataclass
class FlatTrajectory:
    question: str
    raw_output: str                  # full generation with <code>/<answer> tags
    tool_calls: list[dict]           # [{"code": ..., "output": ..., "is_error": ...}, ...]
    final_answer: int | None
    total_tokens: int


class FlatAgent:
    """Frozen base model + one LoRA. Generates reasoning + tool calls in one stream."""

    def __init__(self, base_model, lora_adapter, tool, max_tokens: int = 1024):
        raise NotImplementedError

    def rollout(self, question: str) -> FlatTrajectory:
        """Generate a full trajectory for one problem."""
        raise NotImplementedError
