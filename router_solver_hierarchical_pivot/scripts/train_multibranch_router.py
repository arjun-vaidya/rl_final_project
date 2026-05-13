#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List

from main import load_checkpoint_if_available, load_data, load_model
from src.agents.agent import Agent
from src.memory.hopfield_graph import HeteroHopfieldMemory
from src.routing.simple_router import BRANCH_ORDER, RouterExample, train_multiclass_router
from src.utils.config import get_config
from src.utils.rollout_trace import summarize_rollout_group


def build_agent(model, tokenizer, cfg, branch: str, memory=None) -> Agent:
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
        graph_memory=memory,
        retrieval_top_k=cfg.memory_top_k,
    )


def summarize_branch(agent: Agent, question: str, ground_truth: str, rollouts_per_q: int) -> Dict:
    rollouts = agent.rollout_group(question, ground_truth, rollouts_per_q)
    return summarize_rollout_group(rollouts, ground_truth)


def save_cache(cache_path: str, rows: List[Dict]) -> None:
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cache_path, "w", encoding="ascii", errors="ignore") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)


def load_cache(cache_path: str) -> List[Dict]:
    with open(cache_path, "r", encoding="ascii", errors="ignore") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--train-questions", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--indices-json", default="", help="Optional JSON file containing explicit train indices to evaluate")
    parser.add_argument("--soft-rollouts-per-q", type=int, default=2)
    parser.add_argument("--hard-rollouts-per-q", type=int, default=2)
    parser.add_argument("--memory-trace-input", required=True)
    parser.add_argument("--memory-top-k", type=int, default=2)
    parser.add_argument("--memory-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--memory-embedding-device", default="cpu")
    parser.add_argument("--cache-json", default="", help="Optional branch-summary cache. If present, load instead of regenerating. If absent, write after generation.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    cfg = get_config()
    cfg.dataset_variant = args.dataset
    cfg.train_questions = 400
    cfg.router_temperature = 0.2
    cfg.solver_temperature = 0.7
    cfg.router_max_tokens = 300
    cfg.solver_max_tokens = 512
    cfg.synthesis_max_tokens = 64
    cfg.use_answer_synthesis = False
    cfg.constrained_final_answer_decoding = False
    cfg.plan_parse_repair = True
    cfg.answer_bearing_step_hint = False
    cfg.strict_answer_format = False
    cfg.heuristic_final_selector = False
    cfg.heuristic_final_selector_refined = False
    cfg.guarded_heuristic_fallback = False
    cfg.candidate_rerank = False
    cfg.trace_consistency_guard = False
    cfg.router_prompt_hardening = False
    cfg.memory_top_k = args.memory_top_k
    cfg.memory_embedding_model = args.memory_embedding_model
    cfg.memory_embedding_device = args.memory_embedding_device

    model, tokenizer = load_model(cfg)
    train_qs, train_gts, _test_qs, _test_gts = load_data(cfg)
    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))

    if args.indices_json:
        with open(args.indices_json, "r", encoding="ascii", errors="ignore") as f:
            selected_indices = json.load(f)
        selected_indices = [int(idx) for idx in selected_indices[: args.questions]]
        slice_rows = [(idx, train_qs[idx], train_gts[idx]) for idx in selected_indices]
    else:
        start = args.start_index
        end = start + args.questions
        slice_rows = [(idx, q, gt) for idx, (q, gt) in enumerate(zip(train_qs[start:end], train_gts[start:end]), start=start)]

    examples: List[RouterExample] = []
    audits: List[Dict] = []
    cache_rows: List[Dict] = []
    if args.cache_json and os.path.exists(args.cache_json):
        print(f"[multibranch-router] loading cache from {args.cache_json}", flush=True)
        cache_rows = load_cache(args.cache_json)
    else:
        memory = HeteroHopfieldMemory.from_rollout_traces(
            args.memory_trace_input,
            embedding_model_name=cfg.memory_embedding_model,
            embedding_device=cfg.memory_embedding_device,
        )
        print(f"[multibranch-router] loaded memory cases={memory.num_cases}", flush=True)

        easy_agent = build_agent(model, tokenizer, cfg, branch="easy")
        soft_agent = build_agent(model, tokenizer, cfg, branch="soft")
        hard_agent = build_agent(model, tokenizer, cfg, branch="hard", memory=memory)

        for idx, question, gt in slice_rows:
            print(f"[multibranch-router] q_idx={idx} evaluating easy", flush=True)
            easy_summary = summarize_branch(easy_agent, question, gt, 1)
            print(f"[multibranch-router] q_idx={idx} easy majority={easy_summary['majority_answer']} correct={easy_summary['majority_relaxed_match']}", flush=True)

            print(f"[multibranch-router] q_idx={idx} evaluating soft", flush=True)
            soft_summary = summarize_branch(soft_agent, question, gt, args.soft_rollouts_per_q)
            print(f"[multibranch-router] q_idx={idx} soft majority={soft_summary['majority_answer']} correct={soft_summary['majority_relaxed_match']} support={soft_summary['majority_vote_fraction']:.3f}", flush=True)

            print(f"[multibranch-router] q_idx={idx} evaluating hard", flush=True)
            hard_summary = summarize_branch(hard_agent, question, gt, args.hard_rollouts_per_q)
            print(f"[multibranch-router] q_idx={idx} hard majority={hard_summary['majority_answer']} correct={hard_summary['majority_relaxed_match']} support={hard_summary['majority_vote_fraction']:.3f}", flush=True)

            cache_rows.append(
                {
                    "q_idx": idx,
                    "question": question,
                    "ground_truth": gt,
                    "branch_summaries": {
                        "easy": easy_summary,
                        "soft": soft_summary,
                        "hard": hard_summary,
                    },
                }
            )
        if args.cache_json:
            print(f"[multibranch-router] writing cache to {args.cache_json}", flush=True)
            save_cache(args.cache_json, cache_rows)

    for row in cache_rows:
        example = RouterExample(
            question=row["question"],
            ground_truth=row["ground_truth"],
            branch_summaries=row["branch_summaries"],
        )
        label_name = BRANCH_ORDER[example.label]
        print(f"[multibranch-router] q_idx={row['q_idx']} oracle_label={label_name}", flush=True)
        examples.append(example)
        easy_summary = row["branch_summaries"]["easy"]
        soft_summary = row["branch_summaries"]["soft"]
        hard_summary = row["branch_summaries"]["hard"]
        audits.append(
            {
                "q_idx": row["q_idx"],
                "label": label_name,
                "easy_majority_correct": easy_summary["majority_relaxed_match"],
                "soft_majority_correct": soft_summary["majority_relaxed_match"],
                "hard_majority_correct": hard_summary["majority_relaxed_match"],
                "easy_vote_support": easy_summary["majority_vote_fraction"],
                "soft_vote_support": soft_summary["majority_vote_fraction"],
                "hard_vote_support": hard_summary["majority_vote_fraction"],
            }
        )

    train_examples = examples[: args.train_questions]
    eval_examples = examples[args.train_questions :]
    print(
        f"[multibranch-router] training router train={len(train_examples)} eval={len(eval_examples)}",
        flush=True,
    )
    metrics = train_multiclass_router(train_examples, eval_examples)
    metrics.pop("router")
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "questions": args.questions,
            "train_questions": args.train_questions,
            "start_index": args.start_index,
            "indices_json": args.indices_json,
            "soft_rollouts_per_q": args.soft_rollouts_per_q,
            "hard_rollouts_per_q": args.hard_rollouts_per_q,
            "memory_trace_input": args.memory_trace_input,
            "memory_top_k": args.memory_top_k,
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
