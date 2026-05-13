#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.memory.text_embedder import RetrievalTextEmbedder


def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _lexical_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    a = set(tokens_a)
    b = set(tokens_b)
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--retriever-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    with open(args.corpus_json, "r", encoding="ascii", errors="ignore") as f:
        corpus = json.load(f)
    docs = corpus["docs"]
    queries = corpus["queries"] if args.max_queries <= 0 else corpus["queries"][: args.max_queries]
    device = args.device
    embedder = RetrievalTextEmbedder(args.retriever_dir, device=device, batch_size=args.batch_size)
    doc_texts = [doc.get("doc_text") or doc.get("question", "") for doc in docs]
    query_texts = [query.get("query_text") or query.get("question", "") for query in queries]
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    doc_cache_path = os.path.join(output_dir, "doc_emb_fp16.pt")
    query_cache_path = os.path.join(output_dir, "query_emb_fp16.pt")
    if os.path.exists(doc_cache_path):
        doc_emb = torch.load(doc_cache_path, map_location=device)
    else:
        doc_emb = embedder.encode(doc_texts, role="doc", keep_on_device=True, output_dtype=torch.float16)
        torch.save(doc_emb.detach().cpu(), doc_cache_path)
    if os.path.exists(query_cache_path):
        query_emb = torch.load(query_cache_path, map_location=device)
    else:
        query_emb = embedder.encode(query_texts, role="query", keep_on_device=True, output_dtype=torch.float16)
        torch.save(query_emb.detach().cpu(), query_cache_path)
    sims = torch.matmul(query_emb, doc_emb.T).float()

    doc_tokens = [_tokenize(doc.get("question", "")) for doc in docs]
    doc_by_id: Dict[str, Dict[str, object]] = {doc["doc_id"]: doc for doc in docs}
    positive_doc_ids_by_sig: Dict[str, set] = {}
    for doc in docs:
        if not doc.get("use_as_positive", True):
            continue
        key = f"{doc.get('target_type', 'unknown')}||{doc.get('operation_signature', 'unknown')}"
        positive_doc_ids_by_sig.setdefault(key, set()).add(doc["doc_id"])
    doc_normalized = [doc.get("normalized_question") for doc in docs]
    feature_rows: List[List[float]] = []
    labels: List[float] = []
    top_pool = torch.topk(sims, k=min(max(args.top_k, args.pool_size) + 4, sims.size(1)), dim=1).indices.tolist()
    for q_idx, query in enumerate(queries):
        query_tokens = _tokenize(query.get("question", ""))
        query_target = str(query.get("target_type", "unknown"))
        query_sig = str(query.get("operation_signature", "unknown"))
        positive_ids = positive_doc_ids_by_sig.get(f"{query_target}||{query_sig}", set())
        candidates = []
        for doc_idx in top_pool[q_idx]:
            doc = docs[doc_idx]
            if doc_normalized[doc_idx] == query.get("normalized_question"):
                continue
            candidates.append((float(sims[q_idx, doc_idx].item()), doc_idx))
            if len(candidates) >= max(args.top_k, args.pool_size):
                break
        if not candidates:
            continue
        pool_scores = torch.tensor([score for score, _ in candidates], dtype=torch.float32, device=device)
        pool_emb = torch.stack([doc_emb[idx] for _score, idx in candidates], dim=0).float()
        attn = torch.softmax(args.beta * pool_scores, dim=0)
        hopfield_state = F.normalize((attn.unsqueeze(1) * pool_emb).sum(dim=0), dim=0)
        for pool_rank, (base_score, doc_idx) in enumerate(candidates[: args.top_k]):
            doc = docs[doc_idx]
            hopfield_score = float(torch.dot(hopfield_state, doc_emb[doc_idx].float()).item())
            feature_rows.append([
                float(base_score),
                hopfield_score,
                float(attn[pool_rank].item()),
                float(_lexical_jaccard(query_tokens, doc_tokens[doc_idx])),
                1.0 if str(doc.get("target_type", "unknown")) == query_target else 0.0,
                1.0 if doc.get("use_as_positive", True) else 0.0,
                0.0,
            ])
            labels.append(
                1.0 if doc.get("doc_id") in positive_ids else 0.0
            )

    x = torch.tensor(feature_rows, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    model = nn.Linear(x.size(1), 1, bias=True).to(device)
    positives = float(y.sum().item())
    negatives = float(y.numel() - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _epoch in range(args.epochs):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(1)
        best_threshold = 0.5
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0
        for threshold in [i / 100.0 for i in range(20, 81)]:
            preds = (probs >= threshold).float()
            tp = float(((preds == 1.0) & (y.squeeze(1) == 1.0)).sum().item())
            fp = float(((preds == 1.0) & (y.squeeze(1) == 0.0)).sum().item())
            fn = float(((preds == 0.0) & (y.squeeze(1) == 1.0)).sum().item())
            prec = tp / max(1.0, tp + fp)
            rec = tp / max(1.0, tp + fn)
            f1 = 0.0 if (prec + rec) == 0.0 else (2.0 * prec * rec / (prec + rec))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_prec = prec
                best_rec = rec

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    spec = {
        "feature_names": [
            "base_score",
            "hopfield_score",
            "attention_weight",
            "lexical_jaccard",
            "target_match",
            "quality_score",
            "has_answer_bearing_step",
        ],
        "weights": model.weight.detach().squeeze(0).tolist(),
        "bias": float(model.bias.detach().item()),
        "threshold": float(best_threshold),
        "train_examples": int(x.size(0)),
        "positive_labels": int(y.sum().item()),
        "train_precision": float(best_prec),
        "train_recall": float(best_rec),
        "train_f1": float(best_f1),
        "device": device,
        "top_k": int(args.top_k),
        "pool_size": int(args.pool_size),
        "beta": float(args.beta),
        "retriever_dir": args.retriever_dir,
    }
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(spec, f, indent=2, ensure_ascii=True)
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
