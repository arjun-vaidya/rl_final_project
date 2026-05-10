import json
import os
from typing import Any, Dict, List


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def serialize_rollout(rollout, exact_match: bool, relaxed_match: bool) -> Dict[str, Any]:
    return {
        "valid": rollout.is_valid(),
        "invalid_reason": rollout.invalid_reason,
        "plan": rollout.plan,
        "router_raw_text": rollout.router_raw_text,
        "final_answer": rollout.final_answer,
        "final_answer_source": getattr(rollout, "final_answer_source", None),
        "synthesis_reasoning": getattr(rollout, "synthesis_reasoning", None),
        "candidate_rerank_candidates": getattr(rollout, "candidate_rerank_candidates", []),
        "candidate_rerank_metadata": getattr(rollout, "candidate_rerank_metadata", []),
        "candidate_selector_output": getattr(rollout, "candidate_selector_output", None),
        "answer_bearing_step_idx": getattr(rollout, "answer_bearing_step_idx", None),
        "synthesis_rejected_by_consistency": getattr(rollout, "synthesis_rejected_by_consistency", False),
        "synthesis_vote_answers": getattr(rollout, "synthesis_vote_answers", []),
        "heuristic_selected_candidate": getattr(rollout, "heuristic_selected_candidate", None),
        "exact_match": exact_match,
        "relaxed_match": relaxed_match,
        "router_reward": _safe_float(getattr(rollout, "_router_reward", 0.0)),
        "step_rewards": [_safe_float(x) for x in getattr(rollout, "_step_rewards", [])],
        "outcome_reward": _safe_float(getattr(rollout, "_outcome_reward", 0.0)),
        "steps": [
            {
                "idx": step.idx,
                "subgoal": step.subgoal,
                "answer": step.answer,
                "reasoning": step.reasoning,
            }
            for step in rollout.steps
        ],
    }


def append_rollout_trace(
    trace_path: str,
    epoch: int,
    q_idx: int,
    question: str,
    ground_truth: str,
    rollouts: List[Any],
    exact_matches: List[bool],
    relaxed_matches: List[bool],
) -> None:
    if not trace_path:
        return

    parent = os.path.dirname(trace_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    record = {
        "epoch": epoch,
        "q_idx": q_idx,
        "question": question,
        "ground_truth": ground_truth,
        "rollouts": [
            serialize_rollout(rollout, exact_match, relaxed_match)
            for rollout, exact_match, relaxed_match in zip(rollouts, exact_matches, relaxed_matches)
        ],
    }

    with open(trace_path, "a", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
