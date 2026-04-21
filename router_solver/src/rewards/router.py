"""Router reward.

Spec: docs/04_design.md §"Router reward".
Outcome-gated by structural validity: plan must be valid JSON with 1-8 subgoals.
We deliberately do NOT judge plan quality with an LLM to avoid confounding the experiment.
"""


def is_valid_plan_structure(plan_json_text: str, max_subgoals: int = 8) -> bool:
    """Valid JSON, has 'plan' key, 1 <= len(plan) <= max_subgoals, each item has 'subgoal' + 'tool'."""
    raise NotImplementedError


def router_reward(
    plan_json_text: str,
    trajectory_text: str,
    ground_truth: int,
    max_subgoals: int = 8,
) -> float:
    """0.0 if plan is structurally invalid, else outcome_reward of the trajectory."""
    raise NotImplementedError
