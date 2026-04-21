"""Solver per-step reward.

Spec: docs/04_design.md §"Solver (per step)".
Decomposed: (a) tool executed cleanly → 0.3, (b) output looks sensible → 0.2,
            (c) final trajectory succeeded → 0.5. Weights swept in Ablation A.
"""
from dataclasses import dataclass


@dataclass
class SolverRewardWeights:
    tool_valid: float = 0.3
    sensible_output: float = 0.2
    final_outcome: float = 0.5


def solver_step_reward(
    tool_output: str,
    is_error: bool,
    final_outcome: float,
    weights: SolverRewardWeights = SolverRewardWeights(),
) -> float:
    """Reward for a single Solver step. See spec at top of file."""
    raise NotImplementedError
