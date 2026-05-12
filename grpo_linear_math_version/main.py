import argparse
import json
import logging
import os
import random
import re
import sys

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "linear_reasoning"))

from src.agent import LinearReasoningAgent
from src.config import get_config
from src.eval import evaluate
from src.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
parser.add_argument("--bucket", choices=["mixed", "hard", "trivial", "mixed_hard", "all"], default="all",
                    help="Which probe-derived subset to train on (all = mixed+hard+trivial)")
parser.add_argument("--partition", default="partition.json")
parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
parser.add_argument("--output_dir", default="runs")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--rollouts_per_q", type=int, default=None)
parser.add_argument("--train_microbatch_size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--eval_questions", type=int, default=None)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--shuffle_seed", type=int, default=0, help="Shuffle the filtered train set for variety across epochs")
parser.add_argument("--wandb_project", default="grpo_linear_math")
parser.add_argument("--wandb_run_name", default=None)
parser.add_argument("--no_wandb", action="store_true")
args = parser.parse_args()

cfg = get_config()
cfg.base_model = args.base_model
cfg.output_dir = args.output_dir
if args.rollouts_per_q is not None:
    cfg.rollouts_per_q = args.rollouts_per_q
if args.train_microbatch_size is not None:
    cfg.train_microbatch_size = args.train_microbatch_size
if args.epochs is not None:
    cfg.epochs = args.epochs
if args.learning_rate is not None:
    cfg.learning_rate = args.learning_rate
if args.eval_questions is not None:
    cfg.eval_questions = args.eval_questions

os.makedirs(cfg.output_dir, exist_ok=True)

with open(args.partition) as f:
    partition = json.load(f)

bucket_map = {
    "mixed":      partition["mixed"],
    "hard":       partition["hard"],
    "trivial":    partition.get("trivial", []),
    "mixed_hard": partition["mixed"] + partition["hard"],
    "all":        partition["mixed"] + partition["hard"] + partition.get("trivial", []),
}
train_indices = sorted(set(bucket_map[args.bucket]))

print(f"Base model:    {cfg.base_model}")
print(f"Partition:     {args.partition} (probed_with={partition['probed_with']})")
print(f"Bucket:        {args.bucket}  ->  {len(train_indices)} questions")
print(f"G:             {cfg.rollouts_per_q}  microbatch={cfg.train_microbatch_size}")
print(f"Output dir:    {cfg.output_dir}")


def extract_gt(answer_text):
    last_line = answer_text.split("\n")[-1].strip()
    m = re.search(r"####\s*([-\d.]+)", last_line)
    return m.group(1) if m else answer_text


print("Loading GSM8K...")
ds = load_dataset("gsm8k", "main")
train_q_all = [ex["question"] for ex in ds["train"]]
train_gt_all = [extract_gt(ex["answer"]) for ex in ds["train"]]
test_q = [ex["question"] for ex in ds["test"]]
test_gt = [extract_gt(ex["answer"]) for ex in ds["test"]]

base_q = [train_q_all[i] for i in train_indices]
base_gt = [train_gt_all[i] for i in train_indices]

# Build the multi-epoch training sequence in-place
rng = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else None
train_q, train_gt = [], []
for _ in range(cfg.epochs):
    order = list(range(len(base_q)))
    if rng is not None:
        rng.shuffle(order)
    train_q.extend(base_q[i] for i in order)
    train_gt.extend(base_gt[i] for i in order)

if cfg.eval_questions:
    test_q = test_q[: cfg.eval_questions]
    test_gt = test_gt[: cfg.eval_questions]

print(f"Train: {len(base_q)} unique questions ({args.bucket}) x {cfg.epochs} epochs = {len(train_q)} steps. Test: {len(test_q)} questions.")

if not args.no_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "base_model": cfg.base_model,
            "bucket": args.bucket,
            "n_train": len(train_q),
            "partition_probed_with": partition["probed_with"],
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "rollouts_per_q": cfg.rollouts_per_q,
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

print(f"Loading base model: {cfg.base_model}")
model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
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
    model.print_trainable_parameters()

if args.checkpoint:
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)

agent = LinearReasoningAgent(
    model=model,
    tokenizer=tokenizer,
    max_cot_tokens=cfg.max_cot_tokens,
    temperature=cfg.temperature,
)

if args.mode in ("train", "train_eval"):
    print(f"\n=== TRAINING: {len(base_q)} questions x {cfg.epochs} epochs = {len(train_q)} steps (bucket={args.bucket}) ===")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    train(agent, train_q, train_gt, cfg, optimizer)

if args.mode in ("eval", "train_eval"):
    print(f"\n=== EVALUATING ON {len(test_q)} QUESTIONS (batch_size={args.eval_batch_size}) ===")
    eval_path = os.path.join(cfg.output_dir, "eval_results.json")
    results = evaluate(agent, test_q, test_gt, cfg, output_path=eval_path, batch_size=args.eval_batch_size)
    print(f"\nFinal Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    if wandb.run is not None:
        wandb.log({"eval/accuracy": results["accuracy"], "eval/correct": results["correct"], "eval/total": results["total"]})

if wandb.run is not None:
    wandb.finish()
