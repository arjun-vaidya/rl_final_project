import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from src.utils.answer_utils import clean_answer_text, extract_numeric_value
from src.utils.rollout_trace import append_rollout_trace


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")
ANSWER_LIKE_RE = re.compile(
    r"\b(final|answer|total|altogether|remaining|left|result|cost|sum|difference|product|quotient|how many|how much)\b",
    re.IGNORECASE,
)


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_all_numeric_values(text: Optional[str]) -> List[float]:
    cleaned = clean_answer_text(text)
    if not cleaned:
        return []
    values: List[float] = []
    for match in NUMBER_RE.findall(cleaned.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _contains_value(values: List[float], target: Optional[float], tol: float = 1e-6) -> bool:
    if target is None:
        return False
    return any(abs(value - target) < tol for value in values)


def _is_relaxed_match(answer: Optional[str], ground_truth: Optional[str]) -> bool:
    answer_num = extract_numeric_value(answer)
    gt_num = extract_numeric_value(ground_truth)
    if answer_num is None or gt_num is None:
        return False
    return abs(answer_num - gt_num) < 1e-6


def _shorten(text: Optional[str], limit: int = 180) -> str:
    text = clean_answer_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _classify_rollout(question: str, ground_truth: str, rollout: Dict) -> Dict:
    final_answer = rollout.get("final_answer", "")
    final_num = extract_numeric_value(final_answer)
    gt_num = extract_numeric_value(ground_truth)

    step_answers = [step.get("answer", "") for step in rollout.get("steps", [])]
    step_reasonings = [step.get("reasoning", "") for step in rollout.get("steps", [])]
    step_subgoals = [step.get("subgoal", "") for step in rollout.get("steps", [])]
    step_answer_nums = [_safe_float(extract_numeric_value(answer)) for answer in step_answers]

    all_reasoning_nums: List[float] = []
    for reasoning in step_reasonings:
        all_reasoning_nums.extend(_extract_all_numeric_values(reasoning))

    gt_in_step_answers = _contains_value([x for x in step_answer_nums if x is not None], gt_num)
    gt_in_reasoning = _contains_value(all_reasoning_nums, gt_num)
    correct_number_anywhere = gt_in_step_answers or gt_in_reasoning
    final_reuses_intermediate = _contains_value([x for x in step_answer_nums[:-1] if x is not None], final_num)

    last_subgoal = step_subgoals[-1] if step_subgoals else ""
    last_subgoal_answer_like = bool(ANSWER_LIKE_RE.search(clean_answer_text(last_subgoal)))

    flags = []
    if not rollout.get("valid", False):
        flags.append("invalid_rollout")
    if final_num is None:
        flags.append("non_numeric_final_answer")
    if correct_number_anywhere:
        flags.append("correct_number_appears_in_trace")
    if final_reuses_intermediate:
        flags.append("copied_intermediate_as_final")
    if not last_subgoal_answer_like:
        flags.append("last_subgoal_not_answer_like")

    if not rollout.get("valid", False):
        primary = rollout.get("invalid_reason") or "invalid_rollout"
    elif final_num is None:
        primary = "non_numeric_final_answer"
    elif _is_relaxed_match(final_answer, ground_truth):
        primary = "correct"
    elif correct_number_anywhere:
        primary = "correct_number_in_trace_wrong_final"
    elif final_reuses_intermediate:
        primary = "copied_intermediate_as_final"
    elif not last_subgoal_answer_like:
        primary = "plan_endpoint_mismatch"
    else:
        primary = "wrong_numeric_final"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "primary_category": primary,
        "flags": flags,
        "final_num": final_num,
        "gt_num": gt_num,
        "last_subgoal": last_subgoal,
        "step_answers": step_answers,
        "correct_number_anywhere": correct_number_anywhere,
        "final_reuses_intermediate": final_reuses_intermediate,
        "exact_match": bool(rollout.get("exact_match", False)),
        "relaxed_match": bool(rollout.get("relaxed_match", False)),
        "router_reward": rollout.get("router_reward"),
        "step_rewards": rollout.get("step_rewards", []),
        "outcome_reward": rollout.get("outcome_reward"),
        "last_reasoning": step_reasonings[-1] if step_reasonings else "",
    }


def collect_rollout_traces(agent, questions: List[str], ground_truths: List[str], num_questions: int, rollouts_per_q: int, trace_path: str) -> Dict:
    parent = os.path.dirname(trace_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(trace_path):
        os.remove(trace_path)

    total_questions = 0
    total_rollouts = 0
    valid_rollouts = 0
    exact_correct = 0
    relaxed_correct = 0

    for q_idx, (question, gt) in enumerate(zip(questions[:num_questions], ground_truths[:num_questions])):
        generated_rollouts = agent.rollout_group(question, gt, rollouts_per_q)
        exact_matches = []
        relaxed_matches = []
        valid_count = 0

        for rollout in generated_rollouts:
            exact_match = clean_answer_text(rollout.final_answer).lower() == clean_answer_text(gt).lower()
            relaxed_match = _is_relaxed_match(rollout.final_answer, gt)
            exact_matches.append(exact_match)
            relaxed_matches.append(relaxed_match)
            exact_correct += int(exact_match)
            relaxed_correct += int(relaxed_match)
            if rollout.is_valid():
                valid_count += 1

        append_rollout_trace(
            trace_path,
            epoch=0,
            q_idx=q_idx,
            question=question,
            ground_truth=gt,
            rollouts=generated_rollouts,
            exact_matches=exact_matches,
            relaxed_matches=relaxed_matches,
        )

        total_questions += 1
        total_rollouts += len(generated_rollouts)
        valid_rollouts += valid_count
        print(
            f"Trace Q {q_idx+1}/{num_questions}: valid={valid_count}/{len(generated_rollouts)} | "
            f"relaxed_acc={sum(relaxed_matches)}/{len(generated_rollouts)}"
        )

    return {
        "trace_path": trace_path,
        "questions": total_questions,
        "total_rollouts": total_rollouts,
        "valid_rollouts": valid_rollouts,
        "exact_accuracy": exact_correct / max(total_rollouts, 1),
        "relaxed_numeric_accuracy": relaxed_correct / max(total_rollouts, 1),
    }


def run_trace_taxonomy(trace_path: str, max_failures: int = 50, max_examples_per_category: int = 3) -> Dict:
    records = []
    with open(trace_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_rollouts = 0
    exact_correct = 0
    relaxed_correct = 0
    valid_rollouts = 0
    invalid_counts: Counter = Counter()
    final_answer_source_counts: Counter = Counter()
    primary_counts: Counter = Counter()
    flag_counts: Counter = Counter()
    category_examples: Dict[str, List[Dict]] = defaultdict(list)
    failure_rows: List[Dict] = []

    for record in records:
        question = record.get("question", "")
        ground_truth = record.get("ground_truth", "")
        for rollout in record.get("rollouts", []):
            total_rollouts += 1
            exact_correct += int(bool(rollout.get("exact_match", False)))
            relaxed_correct += int(bool(rollout.get("relaxed_match", False)))
            if rollout.get("valid", False):
                valid_rollouts += 1
            else:
                invalid_counts[rollout.get("invalid_reason") or "invalid_rollout"] += 1
            final_answer_source_counts[rollout.get("final_answer_source") or "none"] += 1

            row = _classify_rollout(question, ground_truth, rollout)
            if row["relaxed_match"]:
                continue

            failure_rows.append(row)
            primary_counts[row["primary_category"]] += 1
            for flag in row["flags"]:
                flag_counts[flag] += 1

            if len(category_examples[row["primary_category"]]) < max_examples_per_category:
                category_examples[row["primary_category"]].append({
                    "question": _shorten(row["question"], 140),
                    "ground_truth": row["ground_truth"],
                    "final_answer": _shorten(row["final_answer"], 120),
                    "last_subgoal": _shorten(row["last_subgoal"], 120),
                    "step_answers": row["step_answers"],
                    "flags": row["flags"],
                    "last_reasoning": _shorten(row["last_reasoning"], 200),
                })

    if max_failures > 0:
        failure_rows = failure_rows[:max_failures]

    return {
        "trace_path": trace_path,
        "records": len(records),
        "total_rollouts": total_rollouts,
        "valid_rollouts": valid_rollouts,
        "exact_accuracy": exact_correct / max(total_rollouts, 1),
        "relaxed_numeric_accuracy": relaxed_correct / max(total_rollouts, 1),
        "num_failures_analyzed": len(failure_rows),
        "invalid_counts": dict(invalid_counts),
        "final_answer_source_counts": dict(final_answer_source_counts),
        "primary_category_counts": dict(primary_counts),
        "flag_counts": dict(flag_counts),
        "examples": dict(category_examples),
        "failure_rows": failure_rows,
    }


def format_taxonomy_report(results: Dict) -> str:
    lines = []
    lines.append("# Failure Taxonomy Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- trace_path: `{results['trace_path']}`")
    lines.append(f"- records: `{results['records']}`")
    lines.append(f"- total_rollouts: `{results['total_rollouts']}`")
    lines.append(f"- valid_rollouts: `{results['valid_rollouts']}`")
    lines.append(f"- exact_accuracy: `{results['exact_accuracy']:.4f}`")
    lines.append(f"- relaxed_numeric_accuracy: `{results['relaxed_numeric_accuracy']:.4f}`")
    lines.append(f"- analyzed_failures: `{results['num_failures_analyzed']}`")
    lines.append(f"- invalid_counts: `{json.dumps(results['invalid_counts'], sort_keys=True)}`")
    lines.append(f"- final_answer_source_counts: `{json.dumps(results['final_answer_source_counts'], sort_keys=True)}`")
    lines.append("")

    lines.append("## Primary Categories")
    if results["primary_category_counts"]:
        for category, count in sorted(results["primary_category_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {category}: `{count}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Flags")
    if results["flag_counts"]:
        for flag, count in sorted(results["flag_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {flag}: `{count}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Example Failures")
    if results["examples"]:
        for category, examples in sorted(results["examples"].items()):
            lines.append(f"### {category}")
            for example in examples:
                lines.append(f"- `{json.dumps(example, ensure_ascii=True)}`")
            lines.append("")
    else:
        lines.append("- none")
        lines.append("")

    return "\n".join(lines)


def save_taxonomy_report(results: Dict, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(format_taxonomy_report(results))
