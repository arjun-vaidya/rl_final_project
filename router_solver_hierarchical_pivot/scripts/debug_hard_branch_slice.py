#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_checkpoint_if_available, load_data, load_model
from src.agents.agent import Agent
from src.agents.vllm_bridge import RouterSolverVLLMBridge, VLLM_AVAILABLE
from src.memory.hopfield_graph import HeteroHopfieldMemory
from src.utils.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-input", default="")
    parser.add_argument("--corpus-json", default="")
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--indices", default="2,3,5")
    parser.add_argument("--memory-top-k", type=int, default=2)
    parser.add_argument("--prompt-top-k", type=int, default=2)
    parser.add_argument("--reranker-path", default="")
    parser.add_argument("--retrieval-gate-threshold", type=float, default=0.0)
    parser.add_argument("--retrieval-gate-coherence", type=float, default=0.0)
    parser.add_argument("--use-hopfield-readout", choices=["on", "off"], default="on")
    parser.add_argument("--use-learned-reranker", choices=["on", "off"], default="on")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--router-max-tokens", type=int, default=300)
    parser.add_argument("--solver-max-tokens", type=int, default=512)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if not args.trace_input and not args.corpus_json:
        raise ValueError("Provide either --trace-input or --corpus-json")

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
    vllm_bridge = None
    if args.use_vllm:
        if not VLLM_AVAILABLE:
            raise RuntimeError("--use-vllm passed but vllm is not importable")
        adapter_root = tempfile.mkdtemp(prefix="router_solver_vllm_")
        router_dir = os.path.join(adapter_root, "router")
        solver_dir = os.path.join(adapter_root, "solver")
        model.save_pretrained(router_dir, selected_adapters=["router"])
        model.save_pretrained(solver_dir, selected_adapters=["solver"])
        vllm_bridge = RouterSolverVLLMBridge(
            base_model=cfg.base_model,
            tokenizer=tokenizer,
            adapter_paths={
                "router": os.path.join(router_dir, "router"),
                "solver": os.path.join(solver_dir, "solver"),
            },
            max_lora_rank=cfg.lora_rank,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )

    indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
    exclude_questions = {train_qs[idx] for idx in indices}
    if args.corpus_json:
        memory = HeteroHopfieldMemory.from_corpus_json(
            args.corpus_json,
            embedding_model_name=args.embedding_model,
            embedding_device=args.embedding_device,
            reranker_path=args.reranker_path,
            retrieval_gate_threshold=args.retrieval_gate_threshold,
            retrieval_gate_coherence=args.retrieval_gate_coherence,
            use_hopfield_readout=args.use_hopfield_readout == "on",
            use_learned_reranker=args.use_learned_reranker == "on",
        )
    else:
        memory = HeteroHopfieldMemory.from_rollout_traces(
            args.trace_input,
            exclude_questions=exclude_questions,
            embedding_model_name=args.embedding_model,
            embedding_device=args.embedding_device,
            reranker_path=args.reranker_path,
            retrieval_gate_threshold=args.retrieval_gate_threshold,
            retrieval_gate_coherence=args.retrieval_gate_coherence,
            use_hopfield_readout=args.use_hopfield_readout == "on",
            use_learned_reranker=args.use_learned_reranker == "on",
        )
    agent = Agent(
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
        execution_branch="hard",
        graph_memory=memory,
        retrieval_top_k=cfg.memory_top_k,
        hard_prompt_top_k=args.prompt_top_k,
        vllm_bridge=vllm_bridge,
    )

    rows = []
    for idx in indices:
        question = train_qs[idx]
        ground_truth = train_gts[idx]
        print(f"[hard-debug] q_idx={idx} start", flush=True)
        rollout = agent.rollout_group(question, ground_truth, 1)[0]
        print(
            f"[hard-debug] q_idx={idx} final={rollout.final_answer} gt={ground_truth} correct={str(rollout.final_answer) == str(ground_truth)}",
            flush=True,
        )
        rows.append(
            {
                "q_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_cases": rollout.retrieved_cases,
                "plan": rollout.plan,
                "final_answer": rollout.final_answer,
                "final_answer_source": rollout.final_answer_source,
                "correct": str(rollout.final_answer) == str(ground_truth),
                "step_answers": [step.answer for step in rollout.steps],
                "steps": [{"idx": step.idx, "subgoal": step.subgoal, "answer": step.answer} for step in rollout.steps],
            }
        )

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump({"indices": indices, "rows": rows}, f, indent=2, ensure_ascii=True)

    print(json.dumps({"indices": indices, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
