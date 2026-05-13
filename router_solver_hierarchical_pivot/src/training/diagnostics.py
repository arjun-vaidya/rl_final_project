import json
import math
import os
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rewards.shaper import RewardShaper
from src.training.train import compute_grpo_advantages
from src.utils.rollout_trace import summarize_rollout_group


def _strict_numeric_match(answer: Optional[str], ground_truth: Optional[str]) -> bool:
    try:
        return abs(float(answer) - float(ground_truth)) < 1e-6
    except Exception:
        return False


def _relaxed_numeric_match(agent, answer: Optional[str], ground_truth: Optional[str]) -> bool:
    answer_num = agent.extract_numeric_value(answer)
    gt_num = agent.extract_numeric_value(ground_truth)
    if answer_num is None or gt_num is None:
        return False
    return abs(answer_num - gt_num) < 1e-6


def _safe_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_std = float(np.std(xs))
    y_std = float(np.std(ys))
    if x_std < 1e-8 or y_std < 1e-8:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _mean_or_zero(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std_or_zero(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _shorten(text: Optional[str], limit: int = 180) -> str:
    text = "" if text is None else str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _assign_rewards(rollouts, shaper):
    judge = shaper.judge

    if judge:
        plan_batch = [(rollout.question, rollout.plan) for rollout in rollouts]
        plan_scores = judge.batch_judge_plans(plan_batch) if plan_batch else []

        step_batch = []
        step_batch_map = {}
        for r_idx, rollout in enumerate(rollouts):
            for step_idx, step in enumerate(rollout.steps):
                step_batch.append((rollout.question, rollout.plan, step_idx, step.reasoning))
                step_batch_map[(r_idx, step_idx)] = len(step_batch) - 1

        step_scores = judge.batch_judge_steps(step_batch) if step_batch else []
    else:
        plan_scores = [min(len(rollout.plan) / 8.0, 1.0) for rollout in rollouts]
        step_scores = []
        step_batch_map = {}
        for r_idx, rollout in enumerate(rollouts):
            for step_idx, step in enumerate(rollout.steps):
                step_batch_map[(r_idx, step_idx)] = len(step_scores)
                step_scores.append(0.3 if len(step.reasoning) > 50 else 0.1)

    for r_idx, rollout in enumerate(rollouts):
        router_reward = plan_scores[r_idx] if r_idx < len(plan_scores) else 0.0
        structural = min(len(rollout.plan) / 8.0, 1.0)
        rollout._router_reward = min(max(0.4 * structural + 0.4 * router_reward + 0.2, 0.0), 1.0)

        step_rewards = []
        for step_idx in range(len(rollout.steps)):
            score_idx = step_batch_map.get((r_idx, step_idx), -1)
            if 0 <= score_idx < len(step_scores):
                step_rewards.append(step_scores[score_idx])
            else:
                step_rewards.append(0.0)
        rollout._step_rewards = step_rewards

        if judge:
            rollout._outcome_reward = judge.judge_answer("", rollout.final_answer, rollout.ground_truth)
        else:
            rollout._outcome_reward = shaper._outcome_reward(rollout.final_answer, rollout.ground_truth)


def deterministic_eval(agent, questions: List[str], ground_truths: List[str], limit: int) -> Dict:
    original_router_temp = agent.router_temperature
    original_solver_temp = agent.solver_temperature
    agent.router_temperature = 0.0
    agent.solver_temperature = 0.0

    results = {
        "total": 0,
        "valid": 0,
        "exact_correct": 0,
        "strict_numeric_correct": 0,
        "relaxed_numeric_correct": 0,
        "plan_lengths": [],
        "invalid_counts": Counter(),
        "audits": [],
    }

    try:
        for question, gt in zip(questions[:limit], ground_truths[:limit]):
            rollout = agent.rollout(question, gt)
            results["total"] += 1

            if not rollout.is_valid():
                results["invalid_counts"][rollout.invalid_reason or "invalid"] += 1
                results["audits"].append({
                    "question": _shorten(question, 120),
                    "ground_truth": gt,
                    "final_answer": rollout.final_answer,
                    "invalid_reason": rollout.invalid_reason or "invalid",
                    "router_raw_text": _shorten(rollout.router_raw_text, 200),
                })
                continue

            results["valid"] += 1
            results["plan_lengths"].append(len(rollout.plan or []))

            exact = bool(rollout.final_answer and rollout.final_answer.strip().lower() == gt.strip().lower())
            strict_numeric = _strict_numeric_match(rollout.final_answer, gt)
            relaxed_numeric = _relaxed_numeric_match(agent, rollout.final_answer, gt)

            results["exact_correct"] += int(exact)
            results["strict_numeric_correct"] += int(strict_numeric)
            results["relaxed_numeric_correct"] += int(relaxed_numeric)

            if not relaxed_numeric:
                results["audits"].append({
                    "question": _shorten(question, 120),
                    "ground_truth": gt,
                    "final_answer": _shorten(rollout.final_answer, 120),
                    "exact": exact,
                    "strict_numeric": strict_numeric,
                    "relaxed_numeric": relaxed_numeric,
                    "last_reasoning": _shorten(rollout.steps[-1].reasoning if rollout.steps else "", 220),
                })
    finally:
        agent.router_temperature = original_router_temp
        agent.solver_temperature = original_solver_temp

    total = max(results["total"], 1)
    results["plan_validity_rate"] = results["valid"] / total
    results["exact_accuracy"] = results["exact_correct"] / total
    results["strict_numeric_accuracy"] = results["strict_numeric_correct"] / total
    results["relaxed_numeric_accuracy"] = results["relaxed_numeric_correct"] / total
    results["avg_plan_len"] = _mean_or_zero(results["plan_lengths"])
    results["audits"] = results["audits"][:10]
    results["invalid_counts"] = dict(results["invalid_counts"])
    return results


def stochastic_group_diagnostics(agent, questions: List[str], ground_truths: List[str], limit: int, rollouts_per_q: int, use_judge: bool) -> Dict:
    shaper = RewardShaper(use_judge=use_judge)
    results = {
        "questions": 0,
        "questions_with_valid_rollouts": 0,
        "valid_rollouts": 0,
        "invalid_counts": Counter(),
        "group_router_stds": [],
        "group_step_stds": [],
        "group_outcome_stds": [],
        "zero_router_adv_groups": 0,
        "zero_step_adv_groups": 0,
        "zero_outcome_adv_groups": 0,
        "router_rewards": [],
        "step_rewards": [],
        "outcome_rewards": [],
        "exact_flags": [],
        "strict_numeric_flags": [],
        "relaxed_numeric_flags": [],
        "majority_exact_flags": [],
        "majority_strict_numeric_flags": [],
        "majority_relaxed_numeric_flags": [],
        "any_relaxed_flags": [],
        "majority_vote_supports": [],
        "audit_failures": [],
    }

    for question, gt in zip(questions[:limit], ground_truths[:limit]):
        results["questions"] += 1
        generated_rollouts = agent.rollout_group(question, gt, rollouts_per_q)
        rollouts = []
        for rollout in generated_rollouts:
            if rollout.is_valid():
                rollouts.append(rollout)
            else:
                results["invalid_counts"][rollout.invalid_reason or "invalid"] += 1

        if not rollouts:
            continue

        results["questions_with_valid_rollouts"] += 1
        results["valid_rollouts"] += len(rollouts)
        _assign_rewards(rollouts, shaper)
        group_summary = summarize_rollout_group(rollouts, gt)
        results["majority_exact_flags"].append(int(group_summary["majority_exact_match"]))
        results["majority_relaxed_numeric_flags"].append(int(group_summary["majority_relaxed_match"]))
        results["any_relaxed_flags"].append(int(group_summary["any_relaxed_match"]))
        results["majority_vote_supports"].append(float(group_summary["majority_vote_fraction"]))
        majority_strict = _strict_numeric_match(group_summary["majority_answer"], gt)
        results["majority_strict_numeric_flags"].append(int(majority_strict))

        group_router = [rollout._router_reward for rollout in rollouts]
        group_step = [sum(rollout._step_rewards) / len(rollout._step_rewards) if rollout._step_rewards else 0.0 for rollout in rollouts]
        group_outcome = [rollout._outcome_reward for rollout in rollouts]

        results["group_router_stds"].append(_std_or_zero(group_router))
        results["group_step_stds"].append(_std_or_zero(group_step))
        results["group_outcome_stds"].append(_std_or_zero(group_outcome))
        results["zero_router_adv_groups"] += int(all(abs(x) < 1e-8 for x in compute_grpo_advantages(group_router)))
        results["zero_step_adv_groups"] += int(all(abs(x) < 1e-8 for x in compute_grpo_advantages(group_step)))
        results["zero_outcome_adv_groups"] += int(all(abs(x) < 1e-8 for x in compute_grpo_advantages(group_outcome)))

        for rollout, step_reward in zip(rollouts, group_step):
            exact = bool(rollout.final_answer and rollout.final_answer.strip().lower() == gt.strip().lower())
            strict_numeric = _strict_numeric_match(rollout.final_answer, gt)
            relaxed_numeric = _relaxed_numeric_match(agent, rollout.final_answer, gt)

            results["router_rewards"].append(rollout._router_reward)
            results["step_rewards"].append(step_reward)
            results["outcome_rewards"].append(rollout._outcome_reward)
            results["exact_flags"].append(int(exact))
            results["strict_numeric_flags"].append(int(strict_numeric))
            results["relaxed_numeric_flags"].append(int(relaxed_numeric))

            if not relaxed_numeric and len(results["audit_failures"]) < 10:
                results["audit_failures"].append({
                    "question": _shorten(question, 120),
                    "ground_truth": gt,
                    "final_answer": _shorten(rollout.final_answer, 120),
                    "group_majority_answer": _shorten(group_summary["majority_answer"], 120),
                    "group_any_relaxed_match": group_summary["any_relaxed_match"],
                    "router_reward": round(float(rollout._router_reward), 4),
                    "step_reward": round(float(step_reward), 4),
                    "outcome_reward": round(float(rollout._outcome_reward), 4),
                    "last_reasoning": _shorten(rollout.steps[-1].reasoning if rollout.steps else "", 220),
                })

    q_total = max(results["questions"], 1)
    valid_rollouts = max(results["valid_rollouts"], 1)
    valid_groups = max(results["questions_with_valid_rollouts"], 1)

    summary = {
        "questions": results["questions"],
        "questions_with_valid_rollouts": results["questions_with_valid_rollouts"],
        "valid_rollout_rate": results["valid_rollouts"] / q_total,
        "avg_router_reward": _mean_or_zero(results["router_rewards"]),
        "avg_step_reward": _mean_or_zero(results["step_rewards"]),
        "avg_outcome_reward": _mean_or_zero(results["outcome_rewards"]),
        "exact_accuracy": sum(results["exact_flags"]) / valid_rollouts,
        "strict_numeric_accuracy": sum(results["strict_numeric_flags"]) / valid_rollouts,
        "relaxed_numeric_accuracy": sum(results["relaxed_numeric_flags"]) / valid_rollouts,
        "majority_exact_accuracy": sum(results["majority_exact_flags"]) / valid_groups,
        "majority_strict_numeric_accuracy": sum(results["majority_strict_numeric_flags"]) / valid_groups,
        "majority_relaxed_numeric_accuracy": sum(results["majority_relaxed_numeric_flags"]) / valid_groups,
        "any_relaxed_accuracy": sum(results["any_relaxed_flags"]) / valid_groups,
        "avg_majority_vote_support": _mean_or_zero(results["majority_vote_supports"]),
        "mean_group_router_std": _mean_or_zero(results["group_router_stds"]),
        "mean_group_step_std": _mean_or_zero(results["group_step_stds"]),
        "mean_group_outcome_std": _mean_or_zero(results["group_outcome_stds"]),
        "zero_router_adv_group_frac": results["zero_router_adv_groups"] / valid_groups,
        "zero_step_adv_group_frac": results["zero_step_adv_groups"] / valid_groups,
        "zero_outcome_adv_group_frac": results["zero_outcome_adv_groups"] / valid_groups,
        "router_reward_relaxed_corr": _safe_corr(results["router_rewards"], results["relaxed_numeric_flags"]),
        "step_reward_relaxed_corr": _safe_corr(results["step_rewards"], results["relaxed_numeric_flags"]),
        "outcome_reward_relaxed_corr": _safe_corr(results["outcome_rewards"], results["relaxed_numeric_flags"]),
        "invalid_counts": dict(results["invalid_counts"]),
        "audit_failures": results["audit_failures"],
    }
    return summary


def run_diagnostics(agent, train_qs: List[str], train_gts: List[str], test_qs: List[str], test_gts: List[str], cfg, diagnostic_questions: int, diagnostic_rollouts_per_q: int) -> Dict:
    return {
        "config": {
            "diagnostic_questions": diagnostic_questions,
            "diagnostic_rollouts_per_q": diagnostic_rollouts_per_q,
            "use_judge": cfg.use_judge,
            "router_temperature": cfg.router_temperature,
            "solver_temperature": cfg.solver_temperature,
        },
        "deterministic_train": deterministic_eval(agent, train_qs, train_gts, diagnostic_questions),
        "deterministic_test": deterministic_eval(agent, test_qs, test_gts, diagnostic_questions),
        "stochastic_train_groups": stochastic_group_diagnostics(
            agent,
            train_qs,
            train_gts,
            diagnostic_questions,
            diagnostic_rollouts_per_q,
            cfg.use_judge,
        ),
    }


def format_diagnostics_report(results: Dict) -> str:
    lines = []
    lines.append("# Training Quality Diagnostics")
    lines.append("")
    lines.append("## Config")
    for key, value in results["config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    for section_key, title in [
        ("deterministic_train", "Deterministic Train Slice"),
        ("deterministic_test", "Deterministic Test Slice"),
    ]:
        section = results[section_key]
        lines.append(f"## {title}")
        lines.append(f"- total: `{section['total']}`")
        lines.append(f"- valid: `{section['valid']}`")
        lines.append(f"- plan_validity_rate: `{section['plan_validity_rate']:.3f}`")
        lines.append(f"- exact_accuracy: `{section['exact_accuracy']:.3f}`")
        lines.append(f"- strict_numeric_accuracy: `{section['strict_numeric_accuracy']:.3f}`")
        lines.append(f"- relaxed_numeric_accuracy: `{section['relaxed_numeric_accuracy']:.3f}`")
        lines.append(f"- avg_plan_len: `{section['avg_plan_len']:.2f}`")
        lines.append(f"- invalid_counts: `{json.dumps(section['invalid_counts'], sort_keys=True)}`")
        lines.append("")
        lines.append("### Audit Samples")
        audits = section["audits"] or [{"note": "No audit failures captured"}]
        for audit in audits[:5]:
            lines.append(f"- `{json.dumps(audit, ensure_ascii=True)}`")
        lines.append("")

    section = results["stochastic_train_groups"]
    lines.append("## Stochastic Train Group Diagnostics")
    scalar_keys = [
        "questions",
        "questions_with_valid_rollouts",
        "valid_rollout_rate",
        "avg_router_reward",
        "avg_step_reward",
        "avg_outcome_reward",
        "exact_accuracy",
        "strict_numeric_accuracy",
        "relaxed_numeric_accuracy",
        "majority_exact_accuracy",
        "majority_strict_numeric_accuracy",
        "majority_relaxed_numeric_accuracy",
        "any_relaxed_accuracy",
        "avg_majority_vote_support",
        "mean_group_router_std",
        "mean_group_step_std",
        "mean_group_outcome_std",
        "zero_router_adv_group_frac",
        "zero_step_adv_group_frac",
        "zero_outcome_adv_group_frac",
        "router_reward_relaxed_corr",
        "step_reward_relaxed_corr",
        "outcome_reward_relaxed_corr",
    ]
    for key in scalar_keys:
        value = section[key]
        if isinstance(value, float) and not math.isnan(value):
            lines.append(f"- {key}: `{value:.4f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append(f"- invalid_counts: `{json.dumps(section['invalid_counts'], sort_keys=True)}`")
    lines.append("")
    lines.append("### Failed Rollout Audits")
    audits = section["audit_failures"] or [{"note": "No failed rollout audits captured"}]
    for audit in audits[:5]:
        lines.append(f"- `{json.dumps(audit, ensure_ascii=True)}`")
    lines.append("")
    return "\n".join(lines)


def save_diagnostics_report(results: Dict, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(format_diagnostics_report(results))
