#!/usr/bin/env python3
"""
Linear Reasoning: RLVR + Self-Reflection on GSM8K with GRPO.

Usage:
    python main.py --mode train         # Train on full GSM8K
    python main.py --mode eval          # Evaluate on full test set
    python main.py --mode train_eval    # Train then evaluate
"""

import argparse
import os
import re
import sys
import torch
import logging
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.agent import LinearReasoningAgent
from src.train import train
from src.eval import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_model(cfg):
    print(f"[1/3] Loading base model: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("[2/3] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    if cfg.use_lora:
        print("[3/3] Setting up LoRA adapter")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_gsm8k():
    """Load GSM8K and extract ground truth answers."""
    dataset = load_dataset("gsm8k", "main")

    def extract_gt(answer_text):
        last_line = answer_text.split("\n")[-1].strip()
        match = re.search(r"####\s*([-\d.]+)", last_line)
        return match.group(1) if match else answer_text

    train_questions = [ex["question"] for ex in dataset["train"]]
    train_gts = [extract_gt(ex["answer"]) for ex in dataset["train"]]
    test_questions = [ex["question"] for ex in dataset["test"]]
    test_gts = [extract_gt(ex["answer"]) for ex in dataset["test"]]

    return train_questions, train_gts, test_questions, test_gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to load")
    parser.add_argument("--train_questions", type=int, default=None, help="Override config (None = full)")
    parser.add_argument("--eval_questions", type=int, default=None, help="Override config (None = full)")
    parser.add_argument("--rollouts_per_q", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--train_microbatch_size", type=int, default=None,
                       help="Trajectories per forward/backward during the policy update")
    parser.add_argument("--wandb_project", default="linear_reasoning", help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg = get_config()
    if args.train_questions is not None:
        cfg.train_questions = args.train_questions
    if args.eval_questions is not None:
        cfg.eval_questions = args.eval_questions
    if args.rollouts_per_q is not None:
        cfg.rollouts_per_q = args.rollouts_per_q
    if args.train_microbatch_size is not None:
        cfg.train_microbatch_size = args.train_microbatch_size

    print(f"\nConfig: train_questions={cfg.train_questions or 'all'}, "
          f"eval_questions={cfg.eval_questions or 'all'}, "
          f"G={cfg.rollouts_per_q}")

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "base_model": cfg.base_model,
                "lora_rank": cfg.lora_rank,
                "lora_alpha": cfg.lora_alpha,
                "rollouts_per_q": cfg.rollouts_per_q,
                "train_questions": cfg.train_questions,
                "eval_questions": cfg.eval_questions,
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "temperature": cfg.temperature,
                "max_cot_tokens": cfg.max_cot_tokens,
                "kl_coef": cfg.kl_coef,
                "correct_reward": cfg.correct_reward,
                "format_reward": cfg.format_reward,
                "train_microbatch_size": cfg.train_microbatch_size,
            },
        )

    # Load model
    model, tokenizer = load_model(cfg)

    # Resume from checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    # Build agent
    agent = LinearReasoningAgent(
        model=model,
        tokenizer=tokenizer,
        max_cot_tokens=cfg.max_cot_tokens,
        temperature=cfg.temperature,
    )

    # Load data
    print("Loading GSM8K...")
    train_q, train_gt, test_q, test_gt = load_gsm8k()
    print(f"GSM8K: {len(train_q)} train, {len(test_q)} test")

    if cfg.train_questions:
        train_q = train_q[:cfg.train_questions]
        train_gt = train_gt[:cfg.train_questions]
    if cfg.eval_questions:
        test_q = test_q[:cfg.eval_questions]
        test_gt = test_gt[:cfg.eval_questions]

    # Train
    if args.mode in ("train", "train_eval"):
        print(f"\n=== TRAINING ON {len(train_q)} QUESTIONS ===")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )
        train(agent, train_q, train_gt, cfg, optimizer)

    # Evaluate
    if args.mode in ("eval", "train_eval"):
        print(f"\n=== EVALUATING ON {len(test_q)} QUESTIONS (batch_size={args.eval_batch_size}) ===")
        eval_path = os.path.join(cfg.output_dir, "eval_results.json")
        results = evaluate(agent, test_q, test_gt, cfg, output_path=eval_path, batch_size=args.eval_batch_size)
        print(f"\nFinal Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
        if wandb.run is not None:
            wandb.log({"eval/accuracy": results["accuracy"], "eval/correct": results["correct"], "eval/total": results["total"]})

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
