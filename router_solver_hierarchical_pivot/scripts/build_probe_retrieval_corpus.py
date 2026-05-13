#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import Counter


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def infer_target_type(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["altogether", "total", "in all", "sum", "combined"]):
        return "total"
    if any(token in lowered for token in ["left", "remain", "remaining", "still need", "more money", "difference", "how many more"]):
        return "difference_or_remaining"
    if any(token in lowered for token in ["percent", "%", "half", "twice", "double", "triple"]):
        return "ratio_or_percent"
    return "direct_answer"


def infer_operation_signature(question: str, reasoning: str) -> str:
    text = f"{question}\n{reasoning}"
    lowered = text.lower()
    tags = []
    if any(token in lowered for token in ["add", "sum", "total", "altogether", "combined", "in all", "+"]):
        tags.append("add")
    if any(token in lowered for token in ["subtract", "difference", "left", "remaining", "more", "-"]):
        tags.append("sub")
    if any(token in lowered for token in ["multiply", "times", "double", "triple", "twice", "product"]):
        tags.append("mul")
    if any(token in lowered for token in ["divide", "half", "percent", "%", "ratio", "quarter"]):
        tags.append("div_ratio")
    if not tags:
        tags.append("direct")
    return "|".join(sorted(set(tags)))


def extract_reasoning_lines(text: str) -> list[str]:
    raw_lines = [line.strip() for line in str(text or "").splitlines()]
    lines = []
    for line in raw_lines:
        if not line:
            continue
        if line.startswith("\\[") or line.startswith("\\("):
            continue
        if line.lower().startswith("thus") or line.lower().startswith("therefore"):
            continue
        lines.append(line)
    if not lines:
        sentence_chunks = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        lines = [chunk.strip() for chunk in sentence_chunks if chunk.strip()]
    return lines[:4]


def build_doc_text(question: str, target_type: str, reasoning_lines: list[str], final_answer: str) -> str:
    body = "\n".join(f"- {line}" for line in reasoning_lines[:4])
    return (
        f"Question: {question}\n"
        f"Target type: {target_type}\n"
        f"Reasoning pattern:\n{body}\n"
        f"Final answer: {final_answer}"
    )


def build_queries(question: str, target_type: str, operation_signature: str, reasoning_lines: list[str]) -> list[str]:
    queries = [question]
    if reasoning_lines:
        queries.append(
            f"Find a solved math example with target type {target_type} and operation signature {operation_signature}. "
            f"Reasoning pattern: {' | '.join(reasoning_lines[:3])}"
        )
    return queries


def extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", str(text or ""))
    if boxed:
        return boxed[-1].strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
    return matches[-1].strip() if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-json", required=True)
    parser.add_argument("--probe-rollouts-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--positive-buckets", default="mixed,hard")
    parser.add_argument("--negative-buckets", default="mixed,hard,trivial")
    parser.add_argument("--max-positive-rollouts-per-question", type=int, default=2)
    parser.add_argument("--max-negative-rollouts-per-question", type=int, default=2)
    args = parser.parse_args()

    with open(args.partition_json, "r", encoding="ascii", errors="ignore") as f:
        partition = json.load(f)

    idx_to_bucket = {}
    for bucket in ["mixed", "hard", "trivial"]:
        for idx in partition.get(bucket, []):
            idx_to_bucket[int(idx)] = bucket

    positive_buckets = {part.strip() for part in args.positive_buckets.split(",") if part.strip()}
    negative_buckets = {part.strip() for part in args.negative_buckets.split(",") if part.strip()}

    docs = []
    queries = []
    stats = Counter()
    positive_by_bucket = Counter()
    negative_by_bucket = Counter()
    signature_counts = Counter()
    doc_counter = 0
    query_counter = 0

    with open(args.probe_rollouts_jsonl, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            q_idx = int(record["idx"])
            bucket = idx_to_bucket.get(q_idx)
            if bucket is None:
                continue
            question = str(record["question"])
            norm_question = normalize_text(question)
            target_type = infer_target_type(question)

            positive_used = 0
            negative_used = 0
            for rollout_idx, rollout in enumerate(record.get("rollouts", [])):
                text = str(rollout.get("text", ""))
                correct = bool(rollout.get("correct", False))
                if correct and bucket in positive_buckets and positive_used >= args.max_positive_rollouts_per_question:
                    continue
                if (not correct) and bucket in negative_buckets and negative_used >= args.max_negative_rollouts_per_question:
                    continue
                if correct:
                    use_as_positive = bucket in positive_buckets
                else:
                    use_as_positive = False
                if not use_as_positive and bucket not in negative_buckets:
                    continue

                reasoning_lines = extract_reasoning_lines(text)
                final_answer = extract_final_answer(text)
                operation_signature = infer_operation_signature(question, "\n".join(reasoning_lines))
                doc_id = f"probe_doc_{doc_counter}"
                doc_counter += 1
                docs.append(
                    {
                        "doc_id": doc_id,
                        "question": question,
                        "normalized_question": norm_question,
                        "target_type": target_type,
                        "operation_signature": operation_signature,
                        "plan": reasoning_lines,
                        "doc_text": build_doc_text(question, target_type, reasoning_lines, final_answer),
                        "trace_path": args.probe_rollouts_jsonl,
                        "source_q_idx": q_idx,
                        "source_rollout_idx": rollout_idx,
                        "final_answer": final_answer,
                        "difficulty_bucket": bucket,
                        "use_as_positive": use_as_positive,
                        "quality_reason": "probe_correct" if use_as_positive else "probe_negative",
                    }
                )
                stats["docs"] += 1
                if use_as_positive:
                    positive_used += 1
                    stats["positive_docs"] += 1
                    positive_by_bucket[bucket] += 1
                    signature_counts[(target_type, operation_signature)] += 1
                    for query_text in build_queries(question, target_type, operation_signature, reasoning_lines):
                        queries.append(
                            {
                                "query_id": f"probe_query_{query_counter}",
                                "doc_id": doc_id,
                                "question": question,
                                "normalized_question": norm_question,
                                "target_type": target_type,
                                "operation_signature": operation_signature,
                                "query_text": query_text,
                            }
                        )
                        query_counter += 1
                else:
                    negative_used += 1
                    stats["negative_docs"] += 1
                    negative_by_bucket[bucket] += 1

    result = {
        "partition_json": args.partition_json,
        "probe_rollouts_jsonl": args.probe_rollouts_jsonl,
        "num_docs": len(docs),
        "num_queries": len(queries),
        "num_positive_docs": stats["positive_docs"],
        "num_negative_only_docs": stats["negative_docs"],
        "positive_difficulty_counts": dict(positive_by_bucket),
        "negative_difficulty_counts": dict(negative_by_bucket),
        "num_signature_buckets": len(signature_counts),
        "signature_buckets": [
            {
                "target_type": key[0],
                "operation_signature": key[1],
                "positive_docs": value,
            }
            for key, value in sorted(signature_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ],
        "docs": docs,
        "queries": queries,
    }

    parent = os.path.dirname(args.output_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_json, "w", encoding="ascii", errors="ignore") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    print(
        json.dumps(
            {
                "num_docs": result["num_docs"],
                "num_queries": result["num_queries"],
                "num_positive_docs": result["num_positive_docs"],
                "num_negative_only_docs": result["num_negative_only_docs"],
                "positive_difficulty_counts": result["positive_difficulty_counts"],
                "negative_difficulty_counts": result["negative_difficulty_counts"],
                "num_signature_buckets": result["num_signature_buckets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
