#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()


@dataclass
class Example:
    query_text: str
    positive_doc_text: str
    negative_doc_text: str


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


class TrainableRetriever(nn.Module):
    def __init__(self, backbone_model_name: str, projection_dim: int = 256, max_length: int = 256, freeze_backbone: bool = True):
        super().__init__()
        config_path = os.path.join(backbone_model_name, "retriever_config.json")
        resolved_backbone = backbone_model_name
        query_prefix = "query: "
        doc_prefix = "passage: "
        projector_state_path = None
        if os.path.isdir(backbone_model_name) and os.path.exists(config_path):
            with open(config_path, "r", encoding="ascii", errors="ignore") as f:
                cfg = json.load(f)
            resolved_backbone = cfg["backbone_model_name"]
            projection_dim = int(cfg.get("projection_dim", projection_dim))
            max_length = int(cfg.get("max_length", max_length))
            query_prefix = str(cfg.get("query_prefix", query_prefix))
            doc_prefix = str(cfg.get("doc_prefix", doc_prefix))
            projector_state_path = os.path.join(backbone_model_name, "projector.pt")

        self.backbone_model_name = resolved_backbone
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_backbone)
        self.backbone = AutoModel.from_pretrained(resolved_backbone)
        hidden_size = int(self.backbone.config.hidden_size)
        self.projector = nn.Linear(hidden_size, projection_dim, bias=False)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        if projector_state_path and os.path.exists(projector_state_path):
            self.projector.load_state_dict(torch.load(projector_state_path, map_location="cpu"))
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def encode(self, texts: List[str], role: str, device: torch.device) -> torch.Tensor:
        prefix = self.query_prefix if role == "query" else self.doc_prefix
        batch = self.tokenizer(
            [prefix + text for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.backbone(**batch)
        token_embeddings = outputs.last_hidden_state
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        projected = self.projector(pooled)
        return F.normalize(projected, dim=1)


def build_examples(corpus_json: str, negatives_json: str, positive_mode: str) -> List[Example]:
    with open(corpus_json, "r", encoding="ascii", errors="ignore") as f:
        corpus = json.load(f)
    with open(negatives_json, "r", encoding="ascii", errors="ignore") as f:
        negatives = json.load(f)

    doc_lookup = {doc["doc_id"]: doc for doc in corpus["docs"]}
    query_lookup = {query["query_id"]: query for query in corpus["queries"]}
    docs_by_signature = {}
    for doc in corpus["docs"]:
        if not doc.get("use_as_positive", True):
            continue
        signature = doc.get("operation_signature") or infer_operation_signature(doc.get("question", ""), doc.get("plan") or [])
        key = (doc["target_type"], signature)
        docs_by_signature.setdefault(key, []).append(doc)
    examples = []
    for row in negatives["negatives"]:
        query = query_lookup[row["query_id"]]
        if not row["hard_negatives"]:
            continue
        pos_doc = doc_lookup[row["doc_id"]]
        if positive_mode == "structural":
            signature = query.get("operation_signature") or infer_operation_signature(query.get("question", ""), [])
            candidates = [
                doc for doc in docs_by_signature.get((query["target_type"], signature), [])
                if doc["normalized_question"] != query["normalized_question"]
            ]
            if candidates:
                pos_doc = random.choice(candidates)
        neg_doc = doc_lookup[row["hard_negatives"][0]["doc_id"]]
        examples.append(
            Example(
                query_text=query["query_text"],
                positive_doc_text=pos_doc["doc_text"],
                negative_doc_text=neg_doc["doc_text"],
            )
        )
    return examples


def iterate_minibatches(examples: List[Example], batch_size: int):
    shuffled = examples[:]
    random.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--negatives-json", required=True)
    parser.add_argument("--base-model", default="intfloat/e5-small-v2")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="")
    parser.add_argument("--positive-mode", choices=["self", "structural"], default="self")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-project", default="router_solver_hierarchical_pivot")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    examples = build_examples(args.corpus_json, args.negatives_json, positive_mode=args.positive_mode)
    if not examples:
        raise ValueError("No training examples were built")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "corpus_json": args.corpus_json,
                    "negatives_json": args.negatives_json,
                    "base_model": args.base_model,
                    "projection_dim": args.projection_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_length": args.max_length,
                    "device": str(device),
                    "positive_mode": args.positive_mode,
                    "num_examples": len(examples),
                },
            )
        except Exception as e:
            print(f"W&B online init failed ({e}); falling back to offline mode.", flush=True)
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode="offline",
                config={
                    "corpus_json": args.corpus_json,
                    "negatives_json": args.negatives_json,
                    "base_model": args.base_model,
                    "projection_dim": args.projection_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_length": args.max_length,
                    "device": str(device),
                    "positive_mode": args.positive_mode,
                    "num_examples": len(examples),
                },
            )
    model = TrainableRetriever(
        backbone_model_name=args.base_model,
        projection_dim=args.projection_dim,
        max_length=args.max_length,
        freeze_backbone=True,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=args.learning_rate)
    epoch_losses = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in iterate_minibatches(examples, args.batch_size):
            q = [item.query_text for item in batch]
            pos = [item.positive_doc_text for item in batch]
            neg = [item.negative_doc_text for item in batch]

            q_emb = model.encode(q, role="query", device=device)
            pos_emb = model.encode(pos, role="doc", device=device)
            neg_emb = model.encode(neg, role="doc", device=device)

            candidate_emb = torch.cat([pos_emb, neg_emb], dim=0)
            logits = torch.matmul(q_emb, candidate_emb.T) / 0.05
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"[retriever-train] epoch={epoch} loss={avg_loss:.4f}", flush=True)
        if wandb.run is not None:
            wandb.log({"train/epoch": epoch, "train/loss": avg_loss})

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector.pt"))
    with open(os.path.join(args.output_dir, "retriever_config.json"), "w", encoding="ascii", errors="ignore") as f:
        json.dump(
            {
                "backbone_model_name": model.backbone_model_name,
                "projection_dim": args.projection_dim,
                "query_prefix": model.query_prefix,
                "doc_prefix": model.doc_prefix,
                "max_length": args.max_length,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="ascii", errors="ignore") as f:
        json.dump(
            {
                "num_examples": len(examples),
                "base_model": args.base_model,
                "projection_dim": args.projection_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "device": str(device),
                "positive_mode": args.positive_mode,
                "epoch_losses": epoch_losses,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
