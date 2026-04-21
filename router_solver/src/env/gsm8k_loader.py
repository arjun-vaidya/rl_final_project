"""GSM8K data loading.

Spec: see docs/03_dataset.md §Splits.
- Train: 7,473 problems (sampled with replacement during RL).
- Val: 200-problem held-out subset of train, fixed seed, for curves + HP decisions.
- Test: full 1,319-problem GSM8K test. Evaluated once per run.
"""
from typing import Iterator, TypedDict


class GSM8KProblem(TypedDict):
    question: str
    answer: int  # extracted from the "#### N" line


def load_gsm8k_train() -> list[GSM8KProblem]:
    """Load GSM8K train split minus the 200 held-out val problems."""
    raise NotImplementedError


def load_gsm8k_val(n: int = 200, seed: int = 0) -> list[GSM8KProblem]:
    """Deterministic val subset carved out of train."""
    raise NotImplementedError


def load_gsm8k_test() -> list[GSM8KProblem]:
    """Full GSM8K test split (1,319 problems). Do NOT tune on this."""
    raise NotImplementedError


def extract_ground_truth(answer_text: str) -> int:
    """Parse the integer after `####` in a GSM8K answer string."""
    raise NotImplementedError
