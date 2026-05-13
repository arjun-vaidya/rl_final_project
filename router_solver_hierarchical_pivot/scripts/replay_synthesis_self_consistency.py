#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_model, get_config, load_checkpoint_if_available
from src.agents.agent import Agent, Step
from src.utils.answer_utils import clean_answer_text, extract_numeric_value


@dataclass
class ReplayRollout:
    question: str
    ground_truth: str
    plan: List[str]
    steps: List[Step]
    recorded_final_answer: str
    recorded_relaxed_match: bool
    replay_final_answer: str = ""
    replay_relaxed_match: bool = False


def _is_relaxed_match(answer: str, ground_truth: str) -> bool:
    a = extract_numeric_value(answer)
    b = extract_numeric_value(ground_truth)
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def load_replay_rollouts(trace_path: str) -> List[ReplayRollout]:
    rollouts: List[ReplayRollout] = []
    with open(trace_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            question = record["question"]
            ground_truth = record["ground_truth"]
            for rollout in record["rollouts"]:
                if not rollout.get("valid", False):
                    continue
                plan = rollout.get("plan") or []
                steps = []
                for step in rollout.get("steps", []):
                    steps.append(
                        Step(
                            idx=int(step["idx"]),
                            subgoal=step["subgoal"],
                            reasoning=step["reasoning"],
                            answer=step["answer"],
                            prompt_ids=None,
                            completion_ids=None,
                        )
                    )
                if not steps or not plan:
                    continue
                rollouts.append(
                    ReplayRollout(
                        question=question,
                        ground_truth=ground_truth,
                        plan=plan,
                        steps=steps,
                        recorded_final_answer=rollout.get("final_answer", ""),
                        recorded_relaxed_match=bool(rollout.get("relaxed_match", False)),
                    )
                )
    return rollouts


def run_replay(trace_path: str, checkpoint: str, output_path: str, samples: int, mode: str) -> Dict[str, Any]:
    cfg = get_config()
    cfg.use_answer_synthesis = True
    cfg.plan_parse_repair = True
    cfg.router_prompt_hardening = False
    cfg.synthesis_self_consistency = mode == "self_consistency"
    cfg.synthesis_self_consistency_samples = samples
    cfg.candidate_rerank = False
    cfg.heuristic_final_selector = mode == "heuristic_selector"
    cfg.heuristic_final_selector_refined = False
    cfg.guarded_heuristic_fallback = False

    model, tokenizer = load_model(cfg)
    agent = Agent(
        model,
        tokenizer,
        router_max_tokens=cfg.router_max_tokens,
        solver_max_tokens=cfg.solver_max_tokens,
        synthesis_max_tokens=cfg.synthesis_max_tokens,
        router_temperature=0.2,
        solver_temperature=1.0,
        use_answer_synthesis=True,
        synthesis_self_consistency=mode == "self_consistency",
        synthesis_self_consistency_samples=samples,
        heuristic_final_selector=mode == "heuristic_selector",
        plan_parse_repair=True,
    )
    load_checkpoint_if_available(model, None, checkpoint, train_len=1)

    replay_rollouts = load_replay_rollouts(trace_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    recorded_correct = 0
    replay_correct = 0
    improved = 0
    regressed = 0
    changed = 0
    rows = []

    for rr in replay_rollouts:
        prompt = agent._build_synthesis_prompt(rr.question, rr.plan, rr.steps)
        gens = agent._generate_same_prompt(
            prompt,
            num_return_sequences=samples,
            max_tokens=cfg.synthesis_max_tokens,
            temp=0.7,
        )
        votes: Dict[str, int] = {}
        best_answer = ""
        best_votes = -1
        sampled_answers: List[str] = []
        for text, _pid, _cid in gens:
            answer = agent._extract_answer(text)
            sampled_answers.append(answer)
            if answer:
                votes[answer] = votes.get(answer, 0) + 1
                if votes[answer] > best_votes:
                    best_votes = votes[answer]
                    best_answer = answer
        rr.replay_final_answer = best_answer
        rr.replay_relaxed_match = _is_relaxed_match(best_answer, rr.ground_truth)

        recorded_correct += int(rr.recorded_relaxed_match)
        replay_correct += int(rr.replay_relaxed_match)
        if clean_answer_text(rr.recorded_final_answer) != clean_answer_text(best_answer):
            changed += 1
        if (not rr.recorded_relaxed_match) and rr.replay_relaxed_match:
            improved += 1
        if rr.recorded_relaxed_match and (not rr.replay_relaxed_match):
            regressed += 1

        rows.append(
            {
                "question": rr.question,
                "ground_truth": rr.ground_truth,
                "recorded_final_answer": rr.recorded_final_answer,
                "recorded_relaxed_match": rr.recorded_relaxed_match,
                "replay_final_answer": rr.replay_final_answer,
                "replay_relaxed_match": rr.replay_relaxed_match,
                "sampled_answers": sampled_answers,
            }
        )

    summary = {
        "trace_path": trace_path,
        "checkpoint": checkpoint,
        "mode": mode,
        "samples": samples,
        "valid_rollouts": len(replay_rollouts),
        "recorded_relaxed_accuracy": recorded_correct / max(len(replay_rollouts), 1),
        "replay_relaxed_accuracy": replay_correct / max(len(replay_rollouts), 1),
        "changed_answers": changed,
        "improved": improved,
        "regressed": regressed,
        "rows": rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--mode", choices=["self_consistency", "heuristic_selector"], default="self_consistency")
    args = parser.parse_args()

    summary = run_replay(args.trace_input, args.checkpoint, args.output_json, args.samples, args.mode)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
