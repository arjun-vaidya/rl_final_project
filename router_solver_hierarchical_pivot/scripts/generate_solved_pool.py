#!/usr/bin/env python3

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_checkpoint_if_available, load_data, load_model
from src.agents.agent import Agent
from src.utils.answer_utils import clean_answer_text, extract_numeric_value
from src.utils.config import get_config
from src.utils.rollout_trace import append_rollout_trace


def relaxed_match(answer: str, ground_truth: str) -> bool:
    a = extract_numeric_value(answer)
    b = extract_numeric_value(ground_truth)
    return a is not None and b is not None and abs(a - b) < 1e-6


def exact_match(answer: str, ground_truth: str) -> bool:
    return clean_answer_text(answer).lower() == clean_answer_text(ground_truth).lower()


def build_agent(model, tokenizer, branch: str, solver_max_tokens: int, router_max_tokens: int) -> Agent:
    return Agent(
        model,
        tokenizer,
        router_max_tokens=router_max_tokens,
        solver_max_tokens=solver_max_tokens,
        synthesis_max_tokens=64,
        router_temperature=0.2,
        solver_temperature=0.7,
        use_answer_synthesis=False,
        constrained_final_answer_decoding=True,
        answer_bearing_step_hint=True,
        plan_parse_repair=True,
        strict_answer_format=False,
        heuristic_final_selector=False,
        heuristic_final_selector_refined=False,
        guarded_heuristic_fallback=False,
        candidate_rerank=False,
        trace_consistency_guard=False,
        router_prompt_hardening=False,
        execution_branch=branch,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--indices", default="")
    parser.add_argument("--indices-file", default="")
    parser.add_argument("--branches", default="easy,soft")
    parser.add_argument("--router-max-tokens", type=int, default=180)
    parser.add_argument("--solver-max-tokens", type=int, default=192)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = get_config()
    cfg.dataset_variant = args.dataset

    model, tokenizer = load_model(cfg)
    train_qs, train_gts, _test_qs, _test_gts = load_data(cfg)
    load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))

    branches = [part.strip() for part in args.branches.split(",") if part.strip()]
    agents = {
        branch: build_agent(model, tokenizer, branch, args.solver_max_tokens, args.router_max_tokens)
        for branch in branches
    }

    for branch in branches:
        trace_path = os.path.join(args.output_dir, f"{branch}_rollout_traces.jsonl")
        if os.path.exists(trace_path):
            os.remove(trace_path)

    if args.indices_file:
        with open(args.indices_file, "r", encoding="ascii", errors="ignore") as f:
            raw = f.read()
        chosen_indices = [int(part.strip()) for part in raw.replace("\n", ",").split(",") if part.strip()]
    elif args.indices:
        chosen_indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
    else:
        end = min(args.start_index + args.questions, len(train_qs))
        chosen_indices = list(range(args.start_index, end))

    for q_idx in chosen_indices:
        if q_idx < 0 or q_idx >= len(train_qs):
            continue
        question = train_qs[q_idx]
        ground_truth = train_gts[q_idx]
        for branch in branches:
            print(f"[solved-pool] q_idx={q_idx} branch={branch}", flush=True)
            rollout = agents[branch].rollout_group(question, ground_truth, 1)[0]
            exacts = [exact_match(rollout.final_answer or "", ground_truth)]
            relaxeds = [relaxed_match(rollout.final_answer or "", ground_truth)]
            append_rollout_trace(
                os.path.join(args.output_dir, f"{branch}_rollout_traces.jsonl"),
                epoch=0,
                q_idx=q_idx,
                question=question,
                ground_truth=ground_truth,
                rollouts=[rollout],
                exact_matches=exacts,
                relaxed_matches=relaxeds,
            )


if __name__ == "__main__":
    main()
