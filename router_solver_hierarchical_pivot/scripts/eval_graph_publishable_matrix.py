#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_checkpoint_if_available, load_data, load_model
from src.agents.agent import Agent
from src.memory.hopfield_graph import HeteroHopfieldMemory
from src.training.taxonomy import run_trace_taxonomy
from src.utils.answer_utils import clean_answer_text
from src.utils.config import get_config
from src.utils.rollout_trace import append_rollout_trace


def _exact_match(answer: str, ground_truth: str) -> bool:
    return clean_answer_text(answer).lower() == clean_answer_text(ground_truth).lower()


def _relaxed_match(answer: str, ground_truth: str) -> bool:
    from src.utils.answer_utils import extract_numeric_value
    a = extract_numeric_value(answer)
    g = extract_numeric_value(ground_truth)
    return a is not None and g is not None and abs(a - g) < 1e-6


def _load_indices(partition_json: str, mixed_n: int, hard_n: int) -> List[int]:
    with open(partition_json, "r", encoding="ascii", errors="ignore") as f:
        part = json.load(f)
    return list(part["mixed"][:mixed_n]) + list(part["hard"][:hard_n])


def _make_hard_agent(base_agent_kwargs: Dict, memory, prompt_top_k: int) -> Agent:
    return Agent(
        **base_agent_kwargs,
        execution_branch="hard",
        graph_memory=memory,
        retrieval_top_k=5,
        hard_prompt_top_k=prompt_top_k,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-input", required=True)
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mixed-count", type=int, default=6)
    parser.add_argument("--hard-count", type=int, default=6)
    parser.add_argument("--screen-count", type=int, default=6)
    parser.add_argument("--dataset", choices=["full", "slim"], default="full")
    parser.add_argument("--router-max-tokens", type=int, default=300)
    parser.add_argument("--solver-max-tokens", type=int, default=256)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--retrieval-gate-threshold", type=float, default=0.05)
    parser.add_argument("--retrieval-gate-coherence", type=float, default=0.45)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--prompt-top-k", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = get_config()
    cfg.dataset_variant = args.dataset
    cfg.train_questions = 10000
    cfg.router_temperature = 0.2
    cfg.solver_temperature = 0.7
    cfg.router_max_tokens = args.router_max_tokens
    cfg.solver_max_tokens = args.solver_max_tokens
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
    cfg.memory_top_k = args.memory_top_k

    model, tokenizer = load_model(cfg)
    train_qs, train_gts, _test_qs, _test_gts = load_data(cfg)
    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))

    indices = _load_indices(args.partition_json, args.mixed_count, args.hard_count)
    selected_questions = {train_qs[idx] for idx in indices}
    with open(os.path.join(args.output_dir, "indices.json"), "w", encoding="ascii", errors="ignore") as f:
        json.dump({"indices": indices}, f, indent=2)

    base_memory = HeteroHopfieldMemory.from_rollout_traces(
        args.trace_input,
        exclude_questions=selected_questions,
        embedding_model_name=args.embedding_model,
        embedding_device=args.embedding_device,
        reranker_path=args.reranker_path,
        retrieval_gate_threshold=args.retrieval_gate_threshold,
        retrieval_gate_coherence=args.retrieval_gate_coherence,
        use_hopfield_readout=True,
        use_learned_reranker=True,
    )

    base_agent_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
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
    )

    variants: List[Tuple[str, Agent, bool]] = [
        ("soft", Agent(**base_agent_kwargs, execution_branch="soft"), False),
        ("hard_retriever_only", _make_hard_agent(base_agent_kwargs, base_memory.fork(use_hopfield_readout=False, use_learned_reranker=False, retrieval_gate_threshold=0.0, retrieval_gate_coherence=0.0), args.prompt_top_k), True),
        ("hard_hopfield", _make_hard_agent(base_agent_kwargs, base_memory.fork(use_hopfield_readout=True, use_learned_reranker=False, retrieval_gate_threshold=0.0, retrieval_gate_coherence=0.0), args.prompt_top_k), True),
        ("hard_reranker", _make_hard_agent(base_agent_kwargs, base_memory.fork(use_hopfield_readout=False, use_learned_reranker=True, retrieval_gate_threshold=0.0, retrieval_gate_coherence=0.0), args.prompt_top_k), True),
        ("hard_full", _make_hard_agent(base_agent_kwargs, base_memory.fork(use_hopfield_readout=True, use_learned_reranker=True, retrieval_gate_threshold=args.retrieval_gate_threshold, retrieval_gate_coherence=args.retrieval_gate_coherence), args.prompt_top_k), False),
    ]

    results: Dict[str, Dict] = {}
    screen_indices = indices[: args.screen_count]

    def run_variant(name: str, agent: Agent, screen_only: bool) -> Dict:
        trace_path = os.path.join(args.output_dir, f"{name}_traces.jsonl")
        if os.path.exists(trace_path):
            os.remove(trace_path)
        rows = []
        correct = 0
        total = 0
        screen_correct = 0
        screened = False
        for pos, idx in enumerate(indices):
            question = train_qs[idx]
            gt = train_gts[idx]
            rollout = agent.rollout_group(question, gt, 1)[0]
            exact = _exact_match(rollout.final_answer or "", gt)
            relaxed = _relaxed_match(rollout.final_answer or "", gt)
            correct += int(relaxed)
            total += 1
            if idx in screen_indices:
                screen_correct += int(relaxed)
            append_rollout_trace(trace_path, 0, idx, question, gt, [rollout], [exact], [relaxed])
            rows.append({
                "q_idx": idx,
                "question": question,
                "ground_truth": gt,
                "final_answer": rollout.final_answer,
                "relaxed_correct": relaxed,
                "retrieved_count": len(getattr(rollout, "retrieved_cases", []) or []),
            })
            print(f"[{name}] q_idx={idx} correct={relaxed} final={rollout.final_answer} gt={gt}", flush=True)
            if screen_only and total >= args.screen_count:
                screened = True
                break
        taxonomy = run_trace_taxonomy(trace_path, max_failures=200, max_examples_per_category=2)
        return {
            "trace_path": trace_path,
            "rows": rows,
            "screen_only": screened,
            "screen_correct": screen_correct,
            "correct": correct,
            "total": total,
            "accuracy": correct / max(1, total),
            "taxonomy": taxonomy,
        }

    results["soft"] = run_variant(*variants[0])
    results["hard_full"] = run_variant(*variants[-1])
    soft_screen = sum(1 for row in results["soft"]["rows"][: args.screen_count] if row["relaxed_correct"])
    full_screen = sum(1 for row in results["hard_full"]["rows"][: args.screen_count] if row["relaxed_correct"])

    for name, agent, screen_first in variants[1:4]:
        screened = run_variant(name, agent, True)
        underperform = screened["screen_correct"] + 1 < full_screen and screened["screen_correct"] <= soft_screen
        if underperform:
            screened["early_stopped"] = True
            results[name] = screened
            print(f"[{name}] early-stop after screen: {screened['screen_correct']}/{args.screen_count}", flush=True)
            continue
        results[name] = run_variant(name, agent, False)
        results[name]["early_stopped"] = False

    helped = 0
    hurt = 0
    soft_map = {row["q_idx"]: row["relaxed_correct"] for row in results["soft"]["rows"]}
    full_map = {row["q_idx"]: row["relaxed_correct"] for row in results["hard_full"]["rows"]}
    for idx in set(soft_map) & set(full_map):
        if full_map[idx] and not soft_map[idx]:
            helped += 1
        if soft_map[idx] and not full_map[idx]:
            hurt += 1

    summary = {
        "indices": indices,
        "variants": {
            name: {
                "accuracy": data["accuracy"],
                "correct": data["correct"],
                "total": data["total"],
                "screen_only": data.get("screen_only", False),
                "early_stopped": data.get("early_stopped", False),
                "primary_failure_counts": data["taxonomy"]["primary_category_counts"],
            }
            for name, data in results.items()
        },
        "graph_helped_vs_soft": helped,
        "graph_hurt_vs_soft": hurt,
        "selected_baseline": "hard_full",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="ascii", errors="ignore") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
