#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import load_data
from src.utils.config import get_config


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_step_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^\*+step\s*\d+\*+\s*:\s*", "", text)
    text = re.sub(r"^step\s*\d+\s*:\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_target_type(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["altogether", "total", "in all", "sum", "combined"]):
        return "total"
    if any(token in lowered for token in ["left", "remain", "remaining", "still need", "more money", "difference", "how many more"]):
        return "difference_or_remaining"
    if any(token in lowered for token in ["percent", "%", "half", "twice", "double", "triple"]):
        return "ratio_or_percent"
    return "direct_answer"


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


def resolve_trace_files(trace_glob: str) -> list[str]:
    paths = sorted(glob.glob(trace_glob))
    return [path for path in paths if os.path.isfile(path)]


def load_probe_buckets(probe_summary_csv: str) -> dict[int, str]:
    if not probe_summary_csv or not os.path.isfile(probe_summary_csv):
        return {}
    out = {}
    with open(probe_summary_csv, "r", encoding="ascii", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            n_correct = int(row["n_correct"])
            if n_correct == 0:
                bucket = "hard"
            elif n_correct == int(row["n_rollouts"]):
                bucket = "trivial"
            else:
                bucket = "mixed"
            out[idx] = bucket
    return out


def load_probe_question_buckets(probe_summary_csv: str) -> dict[str, str]:
    idx_buckets = load_probe_buckets(probe_summary_csv)
    if not idx_buckets:
        return {}
    cfg = get_config()
    cfg.dataset_variant = "full"
    train_qs, _train_gts, _test_qs, _test_gts = load_data(cfg)
    out = {}
    for idx, bucket in idx_buckets.items():
        if 0 <= idx < len(train_qs):
            out[normalize_text(train_qs[idx])] = bucket
    return out


def load_excluded_questions(dataset: str, indices: str) -> set[str]:
    if not indices.strip():
        return set()
    cfg = get_config()
    cfg.dataset_variant = dataset
    train_qs, _train_gts, _test_qs, _test_gts = load_data(cfg)
    excluded = set()
    for part in indices.split(","):
        idx = int(part.strip())
        excluded.add(train_qs[idx])
    return excluded


def build_doc_text(question: str, plan: list[str], target_type: str) -> str:
    key_steps = [str(step) for step in plan[:4]]
    return (
        f"Question: {question}\n"
        f"Target type: {target_type}\n"
        f"Plan pattern:\n" + "\n".join(f"- {step}" for step in key_steps)
    )


def build_queries(question: str, plan: list[str], target_type: str) -> list[str]:
    queries = [question]
    if plan:
        queries.append(f"Retrieve a solved example for target type {target_type}. Final step pattern: {plan[-1]}")
        queries.append(f"Find an analogous math problem with plan structure: {' | '.join(plan[:3])}")
    return queries


def is_answer_like_step(text: str) -> bool:
    lowered = normalize_step_text(text)
    if not lowered:
        return False
    if lowered in {"answer", "final answer"}:
        return False
    markers = [
        "final answer",
        "answer the original question",
        "provide the final answer",
        "present the total",
        "present the answer",
        "report the answer",
        "determine how many",
        "determine how much",
        "determine the total",
        "calculate the total",
        "find the total",
        "find how many",
        "find how much",
        "compute the sum",
        "compute the total",
    ]
    return any(marker in lowered for marker in markers)


def assess_plan_quality(plan: list[str], rollout: dict) -> tuple[bool, str]:
    if not plan:
        return False, "empty_plan"
    normalized = [normalize_step_text(step) for step in plan if normalize_step_text(step)]
    if not normalized:
        return False, "empty_normalized_plan"
    suspicious_tokens = ["human", "assistant", "write a python function", "{\"plan", "```"]
    if any(any(token in step for token in suspicious_tokens) for step in normalized):
        return False, "prompt_leakage"
    if any(step in {"answer", "final answer"} for step in normalized):
        return False, "junk_step"
    unique_ratio = len(set(normalized)) / max(len(normalized), 1)
    if len(normalized) >= 4 and unique_ratio < 0.7:
        return False, "duplicated_steps"
    if len(normalized) >= 6 and len(set(normalized[:3])) == 1:
        return False, "collapsed_prefix"
    answer_bearing_idx = rollout.get("answer_bearing_step_idx")
    if answer_bearing_idx is not None:
        try:
            idx = int(answer_bearing_idx)
            if 0 <= idx < len(plan) and is_answer_like_step(plan[idx]):
                return True, "answer_bearing_idx"
        except (TypeError, ValueError):
            pass
    if is_answer_like_step(plan[-1]):
        return True, "answer_like_final_step"
    return True, "structurally_clean_nonanswer_final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--dataset", choices=["full", "slim"], default="slim")
    parser.add_argument("--exclude-indices", default="")
    parser.add_argument("--probe-summary-csv", default="data_probing/probe_summary.csv")
    parser.add_argument("--positive-difficulty-buckets", default="mixed,hard")
    parser.add_argument("--retain-trivial-as-negative-only", choices=["on", "off"], default="on")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    trace_files = resolve_trace_files(args.trace_glob)
    excluded_questions = {normalize_text(q) for q in load_excluded_questions(args.dataset, args.exclude_indices)}
    probe_buckets = load_probe_buckets(args.probe_summary_csv)
    probe_question_buckets = load_probe_question_buckets(args.probe_summary_csv)
    positive_buckets = {part.strip() for part in args.positive_difficulty_buckets.split(",") if part.strip()}

    docs = []
    queries = []
    doc_counter = 0
    query_counter = 0
    difficulty_counts = Counter()
    positive_counts = Counter()
    signature_counts = Counter()
    rejected_counts = Counter()
    trace_positive_counts = Counter()

    for trace_path in trace_files:
        with open(trace_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                question = str(record.get("question", ""))
                norm_question = normalize_text(question)
                if not question or norm_question in excluded_questions:
                    continue
                q_idx = record.get("q_idx")
                difficulty_bucket = probe_question_buckets.get(norm_question)
                if difficulty_bucket is None:
                    difficulty_bucket = probe_buckets.get(int(q_idx), "unknown") if q_idx is not None else "unknown"
                difficulty_counts[difficulty_bucket] += 1
                target_type = infer_target_type(question)
                for rollout_idx, rollout in enumerate(record.get("rollouts", [])):
                    if not rollout.get("valid", False) or not rollout.get("relaxed_match", False):
                        rejected_counts["invalid_or_incorrect"] += 1
                        continue
                    plan = rollout.get("plan") or []
                    ok, quality_reason = assess_plan_quality(plan, rollout)
                    if not ok:
                        rejected_counts[quality_reason] += 1
                        continue
                    use_as_positive = difficulty_bucket in positive_buckets
                    if difficulty_bucket == "trivial" and args.retain_trivial_as_negative_only == "off":
                        rejected_counts["trivial_excluded"] += 1
                        continue
                    doc_id = f"doc_{doc_counter}"
                    doc_counter += 1
                    operation_signature = infer_operation_signature(question, plan)
                    doc_text = build_doc_text(question, plan, target_type)
                    docs.append(
                        {
                            "doc_id": doc_id,
                            "question": question,
                            "normalized_question": norm_question,
                            "target_type": target_type,
                            "operation_signature": operation_signature,
                            "plan": plan,
                            "doc_text": doc_text,
                            "trace_path": trace_path,
                            "source_q_idx": q_idx,
                            "source_rollout_idx": rollout_idx,
                            "final_answer": rollout.get("final_answer"),
                            "difficulty_bucket": difficulty_bucket,
                            "use_as_positive": use_as_positive,
                            "quality_reason": quality_reason,
                        }
                    )
                    if use_as_positive:
                        positive_counts[difficulty_bucket] += 1
                        signature_counts[(target_type, operation_signature)] += 1
                        trace_positive_counts[trace_path] += 1
                        for query_text in build_queries(question, plan, target_type):
                            queries.append(
                                {
                                    "query_id": f"query_{query_counter}",
                                    "doc_id": doc_id,
                                    "question": question,
                                    "normalized_question": norm_question,
                                    "target_type": target_type,
                                    "operation_signature": operation_signature,
                                    "query_text": query_text,
                                }
                            )
                            query_counter += 1

    signature_rows = [
        {
            "target_type": target_type,
            "operation_signature": op_sig,
            "positive_docs": count,
        }
        for (target_type, op_sig), count in sorted(signature_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    result = {
        "trace_glob": args.trace_glob,
        "num_trace_files": len(trace_files),
        "excluded_questions": len(excluded_questions),
        "num_docs": len(docs),
        "num_queries": len(queries),
        "num_positive_docs": sum(1 for doc in docs if doc["use_as_positive"]),
        "num_negative_only_docs": sum(1 for doc in docs if not doc["use_as_positive"]),
        "difficulty_counts": dict(difficulty_counts),
        "positive_difficulty_counts": dict(positive_counts),
        "rejected_counts": dict(rejected_counts),
        "signature_buckets": signature_rows,
        "num_signature_buckets": len(signature_rows),
        "traces_with_positive_docs": sum(1 for count in trace_positive_counts.values() if count > 0),
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
                k: result[k]
                for k in [
                    "num_trace_files",
                    "excluded_questions",
                    "num_docs",
                    "num_positive_docs",
                    "num_negative_only_docs",
                    "num_queries",
                    "num_signature_buckets",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
