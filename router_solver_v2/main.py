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


def load_data():
    """Load GSM8K dataset. Parses ground truth from '#### <answer>' format."""
    from datasets import load_dataset
    import re

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")

    def parse_answer(answer_text):
        """Extract numeric answer from GSM8K format: '#### 42' -> '42'"""
        last_line = answer_text.split("\n")[-1].strip()
        # GSM8K format: #### <number>
        match = re.search(r'####\s*([-\d.]+)', last_line)
        if match:
            return match.group(1)
        return last_line

    train_qs = [ex["question"] for ex in dataset["train"]]
    train_gts = [parse_answer(ex["answer"]) for ex in dataset["train"]]

    test_qs = [ex["question"] for ex in dataset["test"]]
    test_gts = [parse_answer(ex["answer"]) for ex in dataset["test"]]

    print(f"Train: {len(train_qs)} | Test: {len(test_qs)}")
    print(f"Example GT: '{train_gts[0]}'")
    return train_qs, train_gts, test_qs, test_gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to load")
    args = parser.parse_args()

    cfg = get_config()
    cfg.batch_size = 32
    cfg.rollouts_per_q = 4

    print("\n" + "="*70)
    print("Config for the Script")
    print("="*70)
    print(f"Batch: {cfg.batch_size} | Rollouts/Q: {cfg.rollouts_per_q} | Total/batch: {cfg.total_per_batch}")
    print(f"Epochs: {cfg.epochs} | Use judge: {cfg.use_judge}")
    print()

    # Load model
    model, tokenizer = load_model(cfg)

    agent = Agent(model, tokenizer)

    # Load data
    train_qs, train_gts, test_qs, test_gts = load_data()

    # Limit to 400 questions for faster training
    train_qs = train_qs[:400]
    train_gts = train_gts[:400]
    print(f"Limited training to {len(train_qs)} questions")

    # Train
    if args.mode in ["train", "train_eval"]:
        print("\nStarting training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            try:
                ckpt = torch.load(args.checkpoint)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                    if "optimizer" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer"])
                    print(f"Resumed from epoch {ckpt.get('epoch', 0)}, q {ckpt.get('q_idx', 0)}")
                else:
                    model.load_state_dict(ckpt)
                    print("Loaded legacy checkpoint (model only)")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

        train(agent, train_qs, train_gts, cfg, optimizer)

        # Save final model
        final_ckpt = {
            "model": agent.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": cfg.epochs - 1,
            "config": cfg.__dict__,
        }
        torch.save(final_ckpt, "phase4_final.pt")
        print("Training complete. Final checkpoint saved: phase4_final.pt")

    # Eval
    if args.mode in ["eval", "train_eval"]:
        print("\nEvaluating on test set...")
        results = evaluate(agent, test_qs[:500], test_gts[:500], cfg)  # Eval on 500 examples
        print_eval_results(results, "(Test Set)")

    print("\nDone!")


if __name__ == "__main__":
    main()
