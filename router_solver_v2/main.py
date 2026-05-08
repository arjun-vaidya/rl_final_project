#!/usr/bin/env python3
"""
Main script for Phase 4 training + evaluation.

Usage:
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode train_eval
    python main.py --mode train --checkpoint checkpoint_epoch0_q50.pt
"""

import argparse
import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Load environment variables from .env file
load_dotenv()

from src.utils.config import get_config
from src.agents.agent import Agent
from src.training.train import train
from src.training.eval import evaluate, print_eval_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def load_model(cfg):
    """Load base model and setup LoRA."""
    print(f"\n[1/3] Loading base model: {cfg.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Done: Model loaded")

    print(f"[2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    print(f"Done: Tokenizer loaded")

    if cfg.use_lora:
        print(f"[3/3] Setting up LoRA adapters...")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(f"  Created default adapter")

        # Create separate adapters for router and solver
        model.add_adapter("router", lora_config)
        print(f"  Created router adapter")
        model.add_adapter("solver", lora_config)
        print(f"  Created solver adapter")
        print(f"Done: LoRA adapters ready (default, router, solver)")

    return model, tokenizer


def load_data(cfg):
    """Load GSM8K dataset. Parses ground truth from '#### <answer>' format."""
    from datasets import load_dataset
    import re

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")

    def extract_numeric_answer(answer_text):
        """Extract numeric answer from GSM8K format: '#### 42' -> '42'."""
        last_line = answer_text.split("\n")[-1].strip()
        match = re.search(r'####\s*([-\d.]+)', last_line)
        if match:
            return match.group(1)
        return None

    def build_examples(split):
        questions = []
        answers = []
        for ex in split:
            gt = extract_numeric_answer(ex["answer"])
            if gt is None:
                continue
            questions.append(ex["question"])
            answers.append(gt)
        return questions, answers

    train_split = dataset["train"]
    if cfg.dataset_variant == "slim":
        train_split = train_split.select(range(len(train_split) // 8))
        print(f"Using slim train split derived from GSM8K: raw_size={len(train_split)}")

    train_qs, train_gts = build_examples(train_split)

    test_qs, test_gts = build_examples(dataset["test"])

    print(f"Train: {len(train_qs)} | Test: {len(test_qs)} | Variant: {cfg.dataset_variant}")
    print(f"Example GT: '{train_gts[0]}'")
    return train_qs, train_gts, test_qs, test_gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to load")
    parser.add_argument("--train-questions", type=int, default=None, help="Number of training questions to use")
    parser.add_argument("--eval-questions", type=int, default=None, help="Number of eval questions to use")
    parser.add_argument("--dataset", choices=["full", "slim"], default=None, help="Training dataset variant")
    parser.add_argument("--rollouts-per-q", type=int, default=None, help="GRPO group size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint cadence in questions")
    parser.add_argument("--log-every", type=int, default=None, help="Logging cadence in questions")
    parser.add_argument("--router-max-tokens", type=int, default=None, help="Router generation cap")
    parser.add_argument("--solver-max-tokens", type=int, default=None, help="Solver generation cap")
    parser.add_argument("--router-temperature", type=float, default=None, help="Router sampling temperature")
    parser.add_argument("--solver-temperature", type=float, default=None, help="Solver sampling temperature")
    parser.add_argument("--use-judge", choices=["on", "off"], default=None, help="Enable or disable the judge backend")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and final model")
    args = parser.parse_args()

    cfg = get_config()
    if args.train_questions is not None:
        cfg.train_questions = args.train_questions
    if args.eval_questions is not None:
        cfg.eval_questions = args.eval_questions
    if args.dataset is not None:
        cfg.dataset_variant = args.dataset
    if args.rollouts_per_q is not None:
        cfg.rollouts_per_q = args.rollouts_per_q
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.router_max_tokens is not None:
        cfg.router_max_tokens = args.router_max_tokens
    if args.solver_max_tokens is not None:
        cfg.solver_max_tokens = args.solver_max_tokens
    if args.router_temperature is not None:
        cfg.router_temperature = args.router_temperature
    if args.solver_temperature is not None:
        cfg.solver_temperature = args.solver_temperature
    if args.use_judge is not None:
        cfg.use_judge = args.use_judge == "on"
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    print("\n" + "="*70)
    print("Config for the Script")
    print("="*70)
    print(f"Batch: {cfg.batch_size} | Rollouts/Q: {cfg.rollouts_per_q} | Total/batch: {cfg.total_per_batch}")
    print(f"Dataset: {cfg.dataset_variant} | Train questions: {cfg.train_questions} | Eval questions: {cfg.eval_questions}")
    print(f"Epochs: {cfg.epochs} | LR: {cfg.learning_rate} | Use judge: {cfg.use_judge}")
    print(f"Router max tokens: {cfg.router_max_tokens} @ temp {cfg.router_temperature}")
    print(f"Solver max tokens: {cfg.solver_max_tokens} @ temp {cfg.solver_temperature}")
    print(f"Checkpoint every: {cfg.checkpoint_every} | Output dir: {cfg.output_dir}")
    print()

    # Load model
    model, tokenizer = load_model(cfg)

    agent = Agent(
        model,
        tokenizer,
        router_max_tokens=cfg.router_max_tokens,
        solver_max_tokens=cfg.solver_max_tokens,
        router_temperature=cfg.router_temperature,
        solver_temperature=cfg.solver_temperature,
    )

    # Load data
    train_qs, train_gts, test_qs, test_gts = load_data(cfg)

    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    print(f"Limited training to {len(train_qs)} questions")

    resume_epoch = 0
    resume_q_idx = 0

    # Train
    if args.mode in ["train", "train_eval"]:
        print("\nStarting training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            try:
                ckpt = torch.load(args.checkpoint, map_location="cpu")
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                    if "optimizer" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer"])
                    resume_epoch = int(ckpt.get("epoch", 0))
                    resume_q_idx = int(ckpt.get("q_idx", -1)) + 1
                    if resume_q_idx >= len(train_qs):
                        resume_epoch += 1
                        resume_q_idx = 0
                    print(f"Resumed from epoch {resume_epoch}, q {resume_q_idx}")
                else:
                    model.load_state_dict(ckpt)
                    print("Loaded legacy checkpoint (model only)")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        train(
            agent,
            train_qs,
            train_gts,
            cfg,
            optimizer,
            start_epoch=resume_epoch,
            start_q_idx=resume_q_idx,
        )

        # Save final model
        final_ckpt = {
            "model": agent.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": cfg.epochs - 1,
            "config": cfg.__dict__,
        }
        final_ckpt_path = os.path.join(cfg.output_dir, "phase4_final.pt")
        torch.save(final_ckpt, final_ckpt_path)
        print(f"Training complete. Final checkpoint saved: {final_ckpt_path}")

    # Eval
    if args.mode in ["eval", "train_eval"]:
        print("\nEvaluating on test set...")
        results = evaluate(agent, test_qs[:cfg.eval_questions], test_gts[:cfg.eval_questions], cfg)
        print_eval_results(results, "(Test Set)")

    print("\nDone!")


if __name__ == "__main__":
    main()
