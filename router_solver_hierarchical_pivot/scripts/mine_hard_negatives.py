#!/usr/bin/env python3

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.memory.text_embedder import RetrievalTextEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.corpus_json, "r", encoding="ascii", errors="ignore") as f:
        corpus = json.load(f)

    docs = corpus["docs"]
    queries = corpus["queries"]
    embedder = RetrievalTextEmbedder(args.embedding_model, device=args.embedding_device)
    doc_embs = embedder.encode([doc["doc_text"] for doc in docs], role="doc")
    query_embs = embedder.encode([query["query_text"] for query in queries], role="query")
    sims = torch.matmul(query_embs, doc_embs.T)

    negatives = []
    doc_lookup = {doc["doc_id"]: doc for doc in docs}
    for query_idx, query in enumerate(queries):
        scores, indices = torch.topk(sims[query_idx], k=min(args.top_k + 10, len(docs)))
        mined = []
        for score, doc_idx in zip(scores.tolist(), indices.tolist()):
            doc = docs[doc_idx]
            if doc["doc_id"] == query["doc_id"]:
                continue
            if doc["normalized_question"] == query["normalized_question"]:
                continue
            mined.append({"doc_id": doc["doc_id"], "score": float(score), "question": doc["question"]})
            if len(mined) >= args.top_k:
                break
        negatives.append({"query_id": query["query_id"], "doc_id": query["doc_id"], "hard_negatives": mined})

    result = {
        "corpus_json": args.corpus_json,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "top_k": args.top_k,
        "num_queries": len(queries),
        "negatives": negatives,
    }

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps({"num_queries": len(queries), "top_k": args.top_k}, indent=2))


if __name__ == "__main__":
    main()
