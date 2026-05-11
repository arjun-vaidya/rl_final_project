#!/usr/bin/env python3
# Probe per-question difficulty on the GSM8K train split with Qwen-Math.
# Writes probe_rollouts.jsonl (one record per question, with full rollouts) and
# probe_summary.csv (one row per question). Resumable.

import argparse
import csv
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Run from the linear_reasoning project root so `src/` is importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.agent import LinearReasoningAgent
from src.reward import compute_reward
from src.config import Config

# Every question is wrapped in this system message before going to the model.
SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "showing your reasoning clearly. End with the final answer in \\boxed{} format."
)


def load_gsm8k_train():
    ds = load_dataset("gsm8k", "main")["train"]
    questions, ground_truths = [], []
    for ex in ds:
        questions.append(ex["question"])
        m = re.search(r"####\s*([-\d.]+)", ex["answer"].split("\n")[-1])
        ground_truths.append(m.group(1) if m else ex["answer"])
    return questions, ground_truths


def already_done(jsonl_path):
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path) as f:
        return {json.loads(line)["idx"] for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output_dir", default="data_probing")
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--batch_questions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "probe_rollouts.jsonl")
    csv_path = os.path.join(args.output_dir, "probe_summary.csv")

    cfg = Config()
    cfg.base_model = args.base_model
    cfg.use_lora = False
    cfg.temperature = args.temperature
    cfg.max_cot_tokens = args.max_tokens

    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"Loading {cfg.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    agent = LinearReasoningAgent(model, tokenizer,
                                 max_cot_tokens=cfg.max_cot_tokens,
                                 temperature=cfg.temperature)

    print("Loading GSM8K train split...")
    questions, ground_truths = load_gsm8k_train()
    if args.limit:
        questions = questions[:args.limit]
        ground_truths = ground_truths[:args.limit]

    done = already_done(jsonl_path)
    if done:
        print(f"Resuming, skipping {len(done)} already-probed questions")

    todo = [(i, q, gt) for i, (q, gt) in enumerate(zip(questions, ground_truths)) if i not in done]
    # Sort by length so similar-length prompts batch together and left-padding doesn't waste compute.
    todo.sort(key=lambda r: len(r[1].split()))
    print(f"{len(todo)} questions to probe (G={args.G}, batch={args.batch_questions})")

    write_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(["idx", "q_len_words", "gt", "n_rollouts", "n_correct",
                        "acc", "max_reward", "any_boxed"])
    jsonl_f = open(jsonl_path, "a")

    total_correct = 0
    total_done = 0
    pbar = tqdm(total=len(todo), desc="probing", file=sys.stdout, mininterval=5)
    for chunk_start in range(0, len(todo), args.batch_questions):
        chunk = todo[chunk_start:chunk_start + args.batch_questions]

        # Each question is repeated G times so rollout_batch returns G rollouts per question.
        flat_qs = [q for _, q, _ in chunk for _ in range(args.G)]
        flat_gts = [gt for _, _, gt in chunk for _ in range(args.G)]
        trajs = agent.rollout_batch(flat_qs, flat_gts, temperature=cfg.temperature)

        for chunk_i, (idx, q, gt) in enumerate(chunk):
            group = trajs[chunk_i * args.G:(chunk_i + 1) * args.G]
            rollouts = []
            for t in group:
                reward, is_correct, _ = compute_reward(t.text, t.ground_truth, cfg)
                rollouts.append({"text": t.text, "reward": float(reward), "correct": bool(is_correct)})

            n_correct = sum(r["correct"] for r in rollouts)
            max_reward = max(r["reward"] for r in rollouts)
            any_boxed = any("\\boxed{" in r["text"] for r in rollouts)
            acc = n_correct / args.G

            jsonl_f.write(json.dumps({
                "idx": idx, "question": q, "gt": gt,
                "acc": acc, "n_correct": n_correct, "n_rollouts": args.G,
                "rollouts": rollouts,
            }) + "\n")
            csv_w.writerow([idx, len(q.split()), gt, args.G, n_correct,
                            f"{acc:.4f}", f"{max_reward:.3f}", any_boxed])

            total_correct += int(acc == 1.0)
            total_done += 1

        jsonl_f.flush()
        csv_f.flush()
        pbar.update(len(chunk))
        pbar.set_postfix(all_correct=f"{total_correct}/{total_done}")
    pbar.close()

    csv_f.close()
    jsonl_f.close()
    print(f"\nDone. {csv_path}, {jsonl_path}")


if __name__ == "__main__":
    main()
