#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.memory.text_embedder import RetrievalTextEmbedder


def reciprocal_rank(ranks):
    return 0.0 if not ranks else 1.0 / min(ranks)


def infer_operation_signature(question: str, plan: list[str]) -> str:
    text = f"{question}\n" + "\n".join(plan)
    lowered = text.lower()
    tags = []
    if any(token in lowered for token in ["add", "sum", "total", "altogether", "combined", "in all"]):
        tags.append("add")
    if any(token in lowered for token in ["subtract", "difference", "left", "remaining", "more"]):
        tags.append("sub")
    if any(token in lowered for token in ["multiply", "times", "double", "triple", "twice"]):
        tags.append("mul")
    if any(token in lowered for token in ["divide", "half", "percent", "%", "ratio"]):
        tags.append("div_ratio")
    if not tags:
        tags.append("direct")
    return "|".join(sorted(set(tags)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.corpus_json, "r", encoding="ascii", errors="ignore") as f:
        corpus = json.load(f)

    docs = corpus["docs"]
    queries = corpus["queries"]
    doc_index = {doc["doc_id"]: i for i, doc in enumerate(docs)}
    positive_doc_ids = {doc["doc_id"] for doc in docs if doc.get("use_as_positive", True)}
    embedder = RetrievalTextEmbedder(args.embedding_model, device=args.embedding_device)
    doc_embs = embedder.encode([doc["doc_text"] for doc in docs], role="doc")
    query_embs = embedder.encode([query["query_text"] for query in queries], role="query")
    sims = torch.matmul(query_embs, doc_embs.T)

    docs_by_question = defaultdict(list)
    docs_by_target = defaultdict(list)
    docs_by_signature = defaultdict(list)
    for doc in docs:
        signature = doc.get("operation_signature") or infer_operation_signature(doc.get("question", ""), doc.get("plan") or [])
        if doc["doc_id"] in positive_doc_ids:
            docs_by_question[doc["normalized_question"]].append(doc["doc_id"])
            docs_by_target[doc["target_type"]].append(doc["doc_id"])
            docs_by_signature[(doc["target_type"], signature)].append(doc["doc_id"])

    same_q_rr = []
    same_sig_rr = []
    same_target_rr = []
    same_sig_recall1 = 0
    same_sig_recall3 = 0
    same_sig_recall5 = 0
    same_sig_count = 0

    for q_idx, query in enumerate(queries):
        order = torch.argsort(sims[q_idx], descending=True).tolist()
        same_question = [
            doc_id for doc_id in docs_by_question[query["normalized_question"]]
            if doc_id != query["doc_id"]
        ]
        same_target = [
            doc_id for doc_id in docs_by_target[query["target_type"]]
            if doc_id != query["doc_id"] and docs[doc_index[doc_id]]["normalized_question"] != query["normalized_question"]
        ]
        same_signature = [
            doc_id for doc_id in docs_by_signature[(
                query["target_type"],
                query.get("operation_signature") or infer_operation_signature(query.get("question", ""), []),
            )]
            if doc_id != query["doc_id"] and docs[doc_index[doc_id]]["normalized_question"] != query["normalized_question"]
        ]

        for positive_set, bucket in [
            (same_question, same_q_rr),
            (same_target, same_target_rr),
            (same_signature, same_sig_rr),
        ]:
            if not positive_set:
                continue
            positive_indices = {doc_index[doc_id] for doc_id in positive_set}
            ranks = [rank + 1 for rank, doc_i in enumerate(order) if doc_i in positive_indices]
            bucket.append(reciprocal_rank(ranks))

        if same_signature:
            same_sig_count += 1
            positive_indices = {doc_index[doc_id] for doc_id in same_signature}
            top = order[:5]
            same_sig_recall1 += int(any(doc_i in positive_indices for doc_i in top[:1]))
            same_sig_recall3 += int(any(doc_i in positive_indices for doc_i in top[:3]))
            same_sig_recall5 += int(any(doc_i in positive_indices for doc_i in top[:5]))

    result = {
        "embedding_model": args.embedding_model,
        "num_docs": len(docs),
        "num_positive_docs": len(positive_doc_ids),
        "num_negative_only_docs": len(docs) - len(positive_doc_ids),
        "num_queries": len(queries),
        "num_positive_signature_buckets": len(docs_by_signature),
        "same_question_pairs": len(same_q_rr),
        "same_target_pairs": len(same_target_rr),
        "same_signature_pairs": len(same_sig_rr),
        "same_question_mrr": sum(same_q_rr) / max(len(same_q_rr), 1),
        "same_target_mrr": sum(same_target_rr) / max(len(same_target_rr), 1),
        "same_signature_mrr": sum(same_sig_rr) / max(len(same_sig_rr), 1),
        "same_signature_recall_at_1": same_sig_recall1 / max(same_sig_count, 1),
        "same_signature_recall_at_3": same_sig_recall3 / max(same_sig_count, 1),
        "same_signature_recall_at_5": same_sig_recall5 / max(same_sig_count, 1),
    }

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
