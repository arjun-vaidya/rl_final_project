"""Evaluation harness for all conditions.

Spec: docs/05_evaluation.md.
Computes primary metric (GSM8K accuracy), secondary metrics (tool-validity rate, steps/episode,
tokens/problem), and the gradient-conflict diagnostic.

Usage:
    python -m src.eval.evaluate --checkpoint path/to/ckpt --split test
"""
from dataclasses import dataclass


@dataclass
class EvalReport:
    accuracy: float
    tool_validity_rate: float
    avg_steps_per_episode: float
    avg_tokens_per_problem: float
    advantage_correlation: float | None      # gradient-conflict diagnostic
    wasted_update_fraction: float | None     # gradient-conflict diagnostic


def evaluate(agent, problems: list, compute_diagnostics: bool = False) -> EvalReport:
    """Run `agent` on `problems`, return an EvalReport."""
    raise NotImplementedError


def main() -> None:
    """CLI entry for evaluating a checkpoint on train/val/test."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
