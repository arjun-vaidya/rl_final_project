#!/usr/bin/env python3

import argparse
import json
import os
import sys
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_solver_v2.src.agents.agent import Agent
from router_solver_v2.src.utils.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_numeric_answer(text):
    if not text:
        return None

    lines = text.split('\n')
    for line in reversed(lines):
        match = re.search(r'[-+]?\d*\.?\d+', line.strip())
        if match:
            try:
                return float(match.group())
            except:
                pass
    return None

def load_baseline_results(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_trained_model(checkpoint_path):
    logging.info(f"Loading trained model from {checkpoint_path}")
    cfg = get_config()

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.add_adapter("router", lora_config)
        model.add_adapter("solver", lora_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer, cfg

def evaluate_trained_model(model, tokenizer, cfg, questions, ground_truths, num_samples=100):
    results = {
        "correct": 0,
        "total": 0,
        "model": "Router-Solver V2",
        "details": []
    }

    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        router_adapter="router",
        solver_adapter="solver",
        use_answer_synthesis=cfg.use_answer_synthesis,
        plan_parse_repair=cfg.plan_parse_repair,
    )

    from tqdm import tqdm
    for question, ground_truth in tqdm(zip(questions[:num_samples], ground_truths[:num_samples])):
        results["total"] += 1

        rollout = agent.rollout(question, ground_truth)

        is_correct = False
        if rollout.final_answer and rollout.ground_truth:
            try:
                pred = float(rollout.final_answer)
                gt = float(rollout.ground_truth)
                is_correct = abs(pred - gt) < 1e-6
            except:
                pass

        if is_correct:
            results["correct"] += 1

        results["details"].append({
            "question": question[:100],
            "ground_truth": ground_truth,
            "predicted": rollout.final_answer if rollout.final_answer else "INVALID",
            "correct": is_correct,
            "plan_valid": rollout.is_valid()
        })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results

def generate_comparison_report(all_baselines, trained_results, output_path):
    # Generate comparison markdown report with all three models.
    general = all_baselines.get("general", {})
    math_spec = all_baselines.get("math_specialized", {})

    # Calculate improvements
    general_to_trained = ((trained_results['accuracy'] - general['accuracy']) * 100)
    math_to_trained = ((trained_results['accuracy'] - math_spec['accuracy']) * 100)

    report = f# # Baseline vs Trained Model Comparison
    #
    # | Model | Accuracy | Count |
    # |-------|----------|-------|
    # | **Qwen2.5-1.5B (general)** | {general['accuracy']:.1%} | {general['correct']}/{general['total']} |
    # | **Qwen2.5-Math-1.5B (specialized)** | {math_spec['accuracy']:.1%} | {math_spec['correct']}/{math_spec['total']} |
    # | **Router-Solver V2 (your RL model)** | {trained_results['accuracy']:.1%} | {trained_results['correct']}/{trained_results['total']} |
    #
    # **Improvements vs baselines:**
    # - General baseline → Router-Solver: **+{general_to_trained:.1f} pts**
    # - Math-specialized baseline → Router-Solver: **+{math_to_trained:.1f} pts**
    #
    # ## Models Evaluated
    #
    # - **Qwen2.5-1.5B-Instruct:** General-purpose model, no math specialization
    # - **Qwen2.5-Math-1.5B-Instruct:** Math-specialized pre-trained model
    # - **Router-Solver V2:** Your hierarchical RL-trained model (outcome-heavy rewards)

    with open(output_path, 'w') as f:
        f.write(report)
    logging.info(f"Comparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results", default="baseline/baseline_results_all.json")
    parser.add_argument("--trained_checkpoint",
                       default="router_solver_v2/experiments/outcome_heavy_50q_remotejudge_20260510_100507/phase4_final.pt")
    parser.add_argument("--output", default="baseline/comparison_report.md")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of test questions (default: all)")
    args = parser.parse_args()

    # Load baseline results (both general and math-specialized)
    if os.path.exists(args.baseline_results):
        logging.info("Loading baseline results from cache...")
        all_baselines = load_baseline_results(args.baseline_results)
    else:
        logging.error(f"Baseline results not found at {args.baseline_results}")
        logging.info("Run: python baseline/eval_all_baselines.py first")
        return

    # Load and evaluate trained model
    if os.path.exists(args.trained_checkpoint):
        model, tokenizer, cfg = load_trained_model(args.trained_checkpoint)

        # Load test data
        dataset = load_dataset("gsm8k", "main")
        test_data = dataset["test"]
        if args.num_questions is None:
            num_q = len(test_data)
        else:
            num_q = min(args.num_questions, len(test_data))
        questions = [test_data[i]["question"] for i in range(num_q)]

        ground_truths = []
        for i in range(num_q):
            answer_text = test_data[i]["answer"]
            last_line = answer_text.split("\n")[-1].strip()
            match = re.search(r'####\s*([-\d.]+)', last_line)
            if match:
                ground_truths.append(match.group(1))
            else:
                ground_truths.append(answer_text)

        logging.info(f"Evaluating trained model on {num_q} questions...")
        trained_results = evaluate_trained_model(model, tokenizer, cfg, questions, ground_truths, num_q)
    else:
        logging.error(f"Trained checkpoint not found at {args.trained_checkpoint}")
        return

    logging.info("Generating comparison report")
    generate_comparison_report(all_baselines, trained_results, args.output)
    logging.info(f"Done! Comparison report at {args.output}")

if __name__ == "__main__":
    main()
