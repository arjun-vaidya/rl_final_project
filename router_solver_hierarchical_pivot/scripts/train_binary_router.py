#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List

from main import load_checkpoint_if_available, load_data, load_model
from src.agents.agent import Agent
from src.routing.simple_router import RouterExample, train_binary_router
from src.utils.config import get_config
from src.utils.rollout_trace import summarize_rollout_group


def build_agent(model, tokenizer, cfg, branch: str) -> Agent:
    return Agent(
        model,
        tokenizer,
        router_max_tokens=cfg.router_max_tokens,
        solver_max_tokens=cfg.solver_max_tokens,
        synthesis_max_tokens=cfg.synthesis_max_tokens,
        router_temperature=cfg.router_temperature,
        solver_temperature=cfg.solver_temperature,
        use_answer_synthesis=cfg.use_answer_synthesis,
        constrained_final_answer_decoding=cfg.constrained_final_answer_decoding,
        candidate_rerank=cfg.candidate_rerank,
        trace_consistency_guard=cfg.trace_consistency_guard,
        answer_bearing_step_hint=cfg.answer_bearing_step_hint,
        heuristic_final_selector=cfg.heuristic_final_selector,
        heuristic_final_selector_refined=cfg.heuristic_final_selector_refined,
        guarded_heuristic_fallback=cfg.guarded_heuristic_fallback,
        synthesis_self_consistency=cfg.synthesis_self_consistency,
        synthesis_self_consistency_samples=cfg.synthesis_self_consistency_samples,
        router_prompt_hardening=cfg.router_prompt_hardening,
        plan_parse_repair=cfg.plan_parse_repair,
        strict_answer_format=cfg.strict_answer_format,
        execution_branch=branch,
    )


def summarize_branch(agent: Agent, question: str, ground_truth: str, rollouts_per_q: int) -> Dict:
    rollouts = agent.rollout_group(question, ground_truth, rollouts_per_q)
    return summarize_rollout_group(rollouts, ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--questions", type=int, default=12)
    parser.add_argument("--train-questions", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--soft-rollouts-per-q", type=int, default=6)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    cfg = get_config()
    cfg.dataset_variant = args.dataset
    cfg.train_questions = 400
    cfg.router_temperature = 0.2
    cfg.solver_temperature = 0.7
    cfg.router_max_tokens = 300
    cfg.solver_max_tokens = 512
    cfg.synthesis_max_tokens = 128
    cfg.use_answer_synthesis = False
    cfg.constrained_final_answer_decoding = True
    cfg.plan_parse_repair = True
    cfg.answer_bearing_step_hint = True
    cfg.strict_answer_format = False
    cfg.heuristic_final_selector = False
    cfg.heuristic_final_selector_refined = False
    cfg.guarded_heuristic_fallback = False
    cfg.candidate_rerank = False
    cfg.trace_consistency_guard = False
    cfg.router_prompt_hardening = False

    model, tokenizer = load_model(cfg)
    train_qs, train_gts, _test_qs, _test_gts = load_data(cfg)
    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))

    easy_agent = build_agent(model, tokenizer, cfg, branch="easy")
    soft_agent = build_agent(model, tokenizer, cfg, branch="soft")

    start = args.start_index
    end = start + args.questions
    slice_qs = train_qs[start:end]
    slice_gts = train_gts[start:end]

    examples: List[RouterExample] = []
    audits: List[Dict] = []
    for idx, (question, gt) in enumerate(zip(slice_qs, slice_gts), start=start):
        print(f"[router-smoke] q_idx={idx} evaluating easy branch", flush=True)
        easy_summary = summarize_branch(easy_agent, question, gt, 1)
        print(
            f"[router-smoke] q_idx={idx} easy majority={easy_summary['majority_answer']} "
            f"correct={easy_summary['majority_relaxed_match']}",
            flush=True,
        )
        print(f"[router-smoke] q_idx={idx} evaluating soft branch", flush=True)
        soft_summary = summarize_branch(soft_agent, question, gt, args.soft_rollouts_per_q)
        print(
            f"[router-smoke] q_idx={idx} soft majority={soft_summary['majority_answer']} "
            f"correct={soft_summary['majority_relaxed_match']} support={soft_summary['majority_vote_fraction']:.3f}",
            flush=True,
        )
        example = RouterExample(
            question=question,
            ground_truth=gt,
            easy_summary=easy_summary,
            soft_summary=soft_summary,
        )
        print(
            f"[router-smoke] q_idx={idx} oracle_label={'soft' if example.label == 1 else 'easy'}",
            flush=True,
        )
        examples.append(example)
        audits.append(
            {
                "q_idx": idx,
                "label": "soft" if example.label == 1 else "easy",
                "easy_majority_correct": easy_summary["majority_relaxed_match"],
                "soft_majority_correct": soft_summary["majority_relaxed_match"],
                "easy_vote_support": easy_summary["majority_vote_fraction"],
                "soft_vote_support": soft_summary["majority_vote_fraction"],
            }
        )

    train_examples = examples[: args.train_questions]
    eval_examples = examples[args.train_questions :]
    print(
        f"[router-smoke] training binary router train={len(train_examples)} eval={len(eval_examples)}",
        flush=True,
    )
    metrics = train_binary_router(train_examples, eval_examples)
    metrics.pop("router")
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "questions": args.questions,
            "train_questions": args.train_questions,
            "start_index": args.start_index,
            "soft_rollouts_per_q": args.soft_rollouts_per_q,
        },
        "metrics": metrics,
        "audits": audits,
    }

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
