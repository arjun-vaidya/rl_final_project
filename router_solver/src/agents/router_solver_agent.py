"""Hierarchical Router-Solver agent.

Conditions #3 and #4 in docs/05_evaluation.md. Architecture: docs/02_approach.md.
One frozen base model + two LoRA adapters (Router, Solver), swapped per call.
Plan memory is optional and additive (see docs/07_plan_memory.md).
"""
from dataclasses import dataclass, field


@dataclass
class Subgoal:
    subgoal: str
    tool: str  # "python" for now


@dataclass
class SolverStep:
    subgoal: Subgoal
    code: str
    tool_output: str
    is_error: bool


@dataclass
class HierTrajectory:
    question: str
    plan_raw: str                    # raw Router generation (may be malformed JSON)
    plan: list[Subgoal]              # parsed; empty list if parse failed
    steps: list[SolverStep]
    final_answer: int | None
    total_tokens: int
    retrieved_plans: list[dict] = field(default_factory=list)  # populated iff memory is enabled


class RouterSolverAgent:
    """Router LoRA + Solver LoRA on a shared frozen base.

    If `memory` is not None, the Router prompt is augmented with top-K retrievals.
    """

    def __init__(
        self,
        base_model,
        router_adapter,
        solver_adapter,
        tool,
        memory=None,                 # Optional[PlanMemory]; see src/memory/
        max_subgoals: int = 6,
        router_max_tokens: int = 256,
        solver_max_tokens: int = 128,
    ):
        raise NotImplementedError

    def plan(self, question: str) -> tuple[str, list[Subgoal], list[dict]]:
        """Run Router to produce a plan. Returns (raw_text, parsed_plan, retrieved_plans)."""
        raise NotImplementedError

    def execute(self, question: str, plan: list[Subgoal]) -> list[SolverStep]:
        """Run Solver over each subgoal in sequence, feeding tool outputs forward."""
        raise NotImplementedError

    def rollout(self, question: str) -> HierTrajectory:
        """Plan + execute + parse answer in one shot."""
        raise NotImplementedError
