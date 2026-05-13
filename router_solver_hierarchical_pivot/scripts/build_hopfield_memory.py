#!/usr/bin/env python3

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_data
from src.memory.hopfield_graph import HeteroHopfieldMemory
from src.utils.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-input", default="")
    parser.add_argument("--corpus-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--probe-question", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--exclude-start-index", type=int, default=0)
    parser.add_argument("--exclude-questions", type=int, default=0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args()

    if not args.trace_input and not args.corpus_json:
        raise ValueError("Provide either --trace-input or --corpus-json")

    exclude_set = set()
    if args.exclude_questions > 0:
        cfg = get_config()
        cfg.dataset_variant = args.dataset
        train_qs, _train_gts, _test_qs, _test_gts = load_data(cfg)
        start = max(args.exclude_start_index, 0)
        end = start + max(args.exclude_questions, 0)
        exclude_set = set(train_qs[start:end])

    if args.corpus_json:
        memory = HeteroHopfieldMemory.from_corpus_json(
            args.corpus_json,
            embedding_model_name=args.embedding_model,
            embedding_device=args.embedding_device,
        )
    else:
        memory = HeteroHopfieldMemory.from_rollout_traces(
            args.trace_input,
            exclude_questions=exclude_set,
            embedding_model_name=args.embedding_model,
            embedding_device=args.embedding_device,
        )
    result = {
        "trace_input": args.trace_input,
        "corpus_json": args.corpus_json,
        "excluded_questions": len(exclude_set),
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "num_cases": memory.num_cases,
        "num_nodes": len(memory.nodes),
        "num_edges": len(memory.edges),
    }
    if args.probe_question:
        result["probe_question"] = args.probe_question
        result["probe_results"] = memory.retrieve(args.probe_question, k=args.top_k)

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
