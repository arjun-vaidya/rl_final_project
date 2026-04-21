"""Outcome reward (used by both flat and hierarchical agents).

Spec: docs/04_design.md §"Outcome reward".
Exact numeric match against the GSM8K ground-truth integer.
"""


def extract_numeric_answer(trajectory_text: str) -> int | None:
    """Pull the final integer answer from a trajectory string.

    For flat agent: content of <answer>...</answer>.
    For hier agent: post-processing of the last Solver step (may need a finalize prompt).
    """
    raise NotImplementedError


def outcome_reward(trajectory_text: str, ground_truth: int) -> float:
    """Returns 1.0 iff extracted answer == ground_truth, else 0.0."""
    raise NotImplementedError
