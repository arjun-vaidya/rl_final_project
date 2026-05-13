import json
import os
from typing import Any, Dict, List
from src.utils.answer_utils import clean_answer_text, extract_numeric_value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _relaxed_numeric_match(answer: str, ground_truth: str) -> bool:
    answer_num = extract_numeric_value(answer)
    gt_num = extract_numeric_value(ground_truth)
    if answer_num is None or gt_num is None:
        return False
    return abs(answer_num - gt_num) < 1e-6


def _canonical_answer(text: str) -> str:
    cleaned = clean_answer_text(text)
    numeric = extract_numeric_value(cleaned)
    if numeric is None:
        return cleaned
    return str(int(numeric)) if float(numeric).is_integer() else str(numeric)


def _candidate_quality_score(rollout: Any, candidate: str) -> float:
    candidate_num = extract_numeric_value(candidate)
    score = 0.0
    for item in getattr(rollout, "candidate_rerank_metadata", []) or []:
        item_num = extract_numeric_value(item.get("candidate"))
        if candidate_num is None or item_num is None or abs(candidate_num - item_num) >= 1e-6:
            continue
        score += float(item.get("score", 0.0))
        score += 0.05 * int(item.get("mentions", 1))
        score += 0.01 * int(item.get("step_idx", 0))
        break

    steps = getattr(rollout, "steps", []) or []
    if steps:
        last_step_num = extract_numeric_value(getattr(steps[-1], "answer", ""))
        if candidate_num is not None and last_step_num is not None and abs(candidate_num - last_step_num) < 1e-6:
            score += 1.0

    final_source = str(getattr(rollout, "final_answer_source", "") or "")
    if final_source == "candidate_rerank":
        score += 0.25
    elif final_source == "last_step":
        score += 0.1

    return score


def _extract_all_numeric_values(text: str) -> List[float]:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return []
    values: List[float] = []
    for chunk in cleaned.replace(",", " ").split():
        numeric = extract_numeric_value(chunk)
        if numeric is not None:
            values.append(float(numeric))
    return values


def _contains_value(values: List[float], target: Any, tol: float = 1e-6) -> bool:
    target_float = _safe_float(target)
    return any(abs(value - target_float) < tol for value in values)


def summarize_rollout_group(rollouts: List[Any], ground_truth: str) -> Dict[str, Any]:
    vote_counts: Dict[str, int] = {}
    quality_scores: Dict[str, float] = {}
    chosen_answer = ""
    chosen_votes = 0
    valid_rollouts = 0
    any_exact = False
    any_relaxed = False
    final_answer_source_counts: Dict[str, int] = {}

    for rollout in rollouts:
        if getattr(rollout, "is_valid", None) and rollout.is_valid():
            valid_rollouts += 1

        final_answer = _canonical_answer(getattr(rollout, "final_answer", "") or "")
        final_source = str(getattr(rollout, "final_answer_source", None) or "none")
        final_answer_source_counts[final_source] = final_answer_source_counts.get(final_source, 0) + 1

        exact_match = bool(final_answer and final_answer.lower() == clean_answer_text(ground_truth).lower())
        relaxed_match = _relaxed_numeric_match(final_answer, ground_truth)
        any_exact = any_exact or exact_match
        any_relaxed = any_relaxed or relaxed_match

        if not final_answer:
            continue
        vote_counts[final_answer] = vote_counts.get(final_answer, 0) + 1
        quality_scores[final_answer] = quality_scores.get(final_answer, 0.0) + _candidate_quality_score(rollout, final_answer)

    if vote_counts:
        chosen_answer = min(
            vote_counts.keys(),
            key=lambda answer: (-vote_counts[answer], -quality_scores.get(answer, 0.0), answer),
        )
        chosen_votes = vote_counts[chosen_answer]

    exact_match = bool(chosen_answer and chosen_answer.lower() == clean_answer_text(ground_truth).lower())
    relaxed_match = _relaxed_numeric_match(chosen_answer, ground_truth)

    return {
        "majority_answer": chosen_answer,
        "majority_vote_count": chosen_votes,
        "majority_vote_fraction": chosen_votes / max(len(rollouts), 1),
        "majority_exact_match": exact_match,
        "majority_relaxed_match": relaxed_match,
        "any_exact_match": any_exact,
        "any_relaxed_match": any_relaxed,
        "valid_rollouts": valid_rollouts,
        "total_rollouts": len(rollouts),
        "final_answer_source_counts": final_answer_source_counts,
    }


def serialize_rollout(rollout, exact_match: bool, relaxed_match: bool) -> Dict[str, Any]:
    steps = rollout.steps or []
    step_answers = [getattr(step, "answer", "") for step in steps]
    step_reasonings = [getattr(step, "reasoning", "") for step in steps]
    gt_num = extract_numeric_value(getattr(rollout, "ground_truth", ""))
    answer_idx = getattr(rollout, "answer_bearing_step_idx", None)
    answer_bearing_step_correct = False
    if answer_idx is not None and 0 <= int(answer_idx) < len(step_answers):
        answer_bearing_step_correct = _relaxed_numeric_match(step_answers[int(answer_idx)], rollout.ground_truth)
    trace_nums: List[float] = []
    for text in step_answers + step_reasonings:
        trace_nums.extend(_extract_all_numeric_values(text))
    correct_number_in_trace = _contains_value(trace_nums, gt_num)
    plan = rollout.plan or []
    last_plan_step = plan[-1] if plan else ""
    plan_endpoint_answer_like = any(
        token in clean_answer_text(last_plan_step).lower()
        for token in ("answer", "final", "original question", "total", "remaining", "left", "how many", "how much")
    )
    return {
        "valid": rollout.is_valid(),
        "invalid_reason": rollout.invalid_reason,
        "plan": plan,
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
        "retrieved_cases": getattr(rollout, "retrieved_cases", []),
        "exact_match": exact_match,
        "relaxed_match": relaxed_match,
        "plan_endpoint_answer_like": plan_endpoint_answer_like,
        "answer_bearing_step_correct": answer_bearing_step_correct,
        "correct_number_in_trace": correct_number_in_trace,
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
        "group_summary": summarize_rollout_group(rollouts, ground_truth),
        "rollouts": [
            serialize_rollout(rollout, exact_match, relaxed_match)
            for rollout, exact_match, relaxed_match in zip(rollouts, exact_matches, relaxed_matches)
        ],
    }

    with open(trace_path, "a", encoding="ascii", errors="ignore") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
