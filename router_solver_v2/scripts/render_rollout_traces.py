#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _rollout_status(rollout: Dict[str, Any]) -> str:
    valid = rollout.get("valid", False)
    exact = rollout.get("exact_match", False)
    relaxed = rollout.get("relaxed_match", False)
    if not valid:
        reason = rollout.get("invalid_reason", "unknown")
        return f"invalid ({reason})"
    return f"valid | exact={exact} | relaxed={relaxed}"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def render_trace(input_path: Path, output_path: Path, max_questions: int = 0) -> None:
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if max_questions > 0:
        rows = rows[:max_questions]

    total_rollouts = 0
    valid_rollouts = 0
    exact_correct = 0
    relaxed_correct = 0

    lines: List[str] = []
    lines.append(f"# Human-Readable Rollout Trace")
    lines.append("")
    lines.append(f"- Source: `{input_path}`")
    lines.append(f"- Questions rendered: `{len(rows)}`")
    lines.append("")

    for qn, row in enumerate(rows, start=1):
        question = _safe_text(row.get("question", ""))
        gt = _safe_text(row.get("ground_truth", ""))
        epoch = row.get("epoch", "n/a")
        q_idx = row.get("q_idx", "n/a")
        rollouts = row.get("rollouts", [])
        q_valid = 0
        q_exact = 0
        q_relaxed = 0

        lines.append(f"## Q{qn} (epoch={epoch}, q_idx={q_idx})")
        lines.append("")
        lines.append("Question:")
        lines.append("")
        lines.append(question)
        lines.append("")
        lines.append(f"Ground truth: `{gt}`")
        lines.append("")
        summary_idx = len(lines)
        lines.append("Rollout summary: pending")
        lines.append("")

        for ridx, rollout in enumerate(rollouts, start=1):
            total_rollouts += 1
            valid = rollout.get("valid", False)
            exact = rollout.get("exact_match", False)
            relaxed = rollout.get("relaxed_match", False)
            if valid:
                valid_rollouts += 1
                q_valid += 1
            if exact:
                exact_correct += 1
                q_exact += 1
            if relaxed:
                relaxed_correct += 1
                q_relaxed += 1

            lines.append(f"### Rollout {ridx}")
            lines.append("")
            lines.append(f"Status: `{_rollout_status(rollout)}`")
            lines.append(f"Final answer source: `{rollout.get('final_answer_source', 'unknown')}`")
            lines.append(f"Final answer: `{_safe_text(rollout.get('final_answer', ''))}`")
            lines.append(f"Router reward: `{_fmt_float(rollout.get('router_reward', 0.0))}`")
            lines.append(f"Outcome reward: `{_fmt_float(rollout.get('outcome_reward', 0.0))}`")

            step_rewards = rollout.get("step_rewards", [])
            if step_rewards:
                lines.append(
                    f"Step rewards: `{', '.join(_fmt_float(x) for x in step_rewards)}`"
                )

            plan = rollout.get("plan", [])
            if plan:
                lines.append("Plan:")
                for i, step in enumerate(plan, start=1):
                    lines.append(f"{i}. {_safe_text(step)}")

            steps = rollout.get("steps", [])
            if steps:
                lines.append("Solver steps:")
                for s in steps:
                    idx = s.get("idx", "?")
                    subgoal = _safe_text(s.get("subgoal", ""))
                    reasoning = _safe_text(s.get("reasoning", ""))
                    answer = _safe_text(s.get("answer", ""))
                    lines.append(f"Step {idx}: {subgoal}")
                    lines.append(f"Extracted answer: `{answer}`")
                    lines.append("Reasoning:")
                    lines.append("```text")
                    lines.append(reasoning)
                    lines.append("```")

            if rollout.get("candidate_rerank_candidates"):
                lines.append(f"Candidate rerank set: `{rollout.get('candidate_rerank_candidates')}`")
            if rollout.get("synthesis_vote_answers"):
                lines.append(f"Synthesis votes: `{rollout.get('synthesis_vote_answers')}`")

            lines.append("")

        lines[summary_idx] = (
            f"Rollout summary: `{q_valid}/{len(rollouts)}` valid | "
            f"exact `{q_exact}/{len(rollouts)}` | relaxed `{q_relaxed}/{len(rollouts)}`"
        )

    lines.insert(
        4,
        f"- Aggregate rollout accuracy (exact/relaxed): `{(exact_correct / max(total_rollouts, 1)):.3f}` / `{(relaxed_correct / max(total_rollouts, 1)):.3f}`",
    )
    lines.insert(
        5,
        f"- Aggregate validity: `{valid_rollouts}/{total_rollouts}` ({(valid_rollouts / max(total_rollouts, 1)):.3f})",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rollout traces JSONL into human-readable markdown.")
    parser.add_argument("--input", required=True, help="Path to rollout_traces.jsonl")
    parser.add_argument("--output", required=True, help="Path to markdown output")
    parser.add_argument("--max-questions", type=int, default=0, help="Optional cap; 0 means all")
    args = parser.parse_args()

    render_trace(Path(args.input), Path(args.output), args.max_questions)


if __name__ == "__main__":
    main()
