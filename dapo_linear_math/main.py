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

ROOT = os.path.dirname(os.path.abspath(__file__))
LINEAR_SRC = os.path.join(ROOT, "..", "linear_reasoning", "src")
DAPO_SRC = os.path.join(ROOT, "src")
sys.path.insert(0, LINEAR_SRC)
sys.path.insert(0, DAPO_SRC)

from agent import LinearReasoningAgent
from config import get_config
from train_dapo import train_dapo
from eval_sc import evaluate_sc
from vllm_rollout import VLLMRollout, VLLM_AVAILABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
parser.add_argument("--bucket", choices=["mixed", "hard", "trivial", "mixed_hard", "all"], default="mixed_hard")
parser.add_argument("--partition", default="partition.json")
parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
parser.add_argument("--output_dir", default="runs")
parser.add_argument("--checkpoint", default=None)

# Core hyperparameters
parser.add_argument("--rollouts_per_q", type=int, default=8)
parser.add_argument("--train_microbatch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--kl_coef", type=float, default=0.015)

# DAPO knobs
parser.add_argument("--dapo_groups_per_step", type=int, default=2,
                    help="Target number of informative groups per gradient step.")
parser.add_argument("--dapo_rollout_batch", type=int, default=2,
                    help="Questions to draw per resample attempt.")
parser.add_argument("--dapo_max_resamples", type=int, default=8,
                    help="Resample cap per step before giving up on this step.")

# LoRA
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=64)
parser.add_argument("--lora_target_all_linear", action="store_true", default=True,
                    help="Target q,k,v,o + gate,up,down (vs the default q,v only).")

# Eval
parser.add_argument("--eval_questions", type=int, default=None)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--sc_K", type=int, default=8, help="Self-consistency samples at eval time.")
parser.add_argument("--sc_temperature", type=float, default=0.6)

# vLLM
parser.add_argument("--use_vllm", action="store_true",
                    help="Use vLLM for rollouts (5-10x faster gen). Requires `pip install vllm`.")
parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45,
                    help="Fraction of GPU memory vLLM may use. Lower if HF training OOMs.")
parser.add_argument("--vllm_sync_every", type=int, default=1,
                    help="Sync the PEFT adapter to vLLM every N gradient steps.")

parser.add_argument("--shuffle_seed", type=int, default=0)
parser.add_argument("--wandb_project", default="dapo_linear_math")
parser.add_argument("--wandb_run_name", default=None)
parser.add_argument("--no_wandb", action="store_true")
args = parser.parse_args()

cfg = get_config()
cfg.base_model = args.base_model
cfg.output_dir = args.output_dir
cfg.rollouts_per_q = args.rollouts_per_q
cfg.train_microbatch_size = args.train_microbatch_size
cfg.epochs = args.epochs
cfg.learning_rate = args.learning_rate
cfg.kl_coef = args.kl_coef
cfg.lora_rank = args.lora_rank
cfg.lora_alpha = args.lora_alpha
if args.eval_questions is not None:
    cfg.eval_questions = args.eval_questions

# DAPO knobs are attached to the config so train_dapo can read them.
cfg.dapo_groups_per_step = args.dapo_groups_per_step
cfg.dapo_rollout_batch = args.dapo_rollout_batch
cfg.dapo_max_resamples = args.dapo_max_resamples
cfg.vllm_sync_every = args.vllm_sync_every

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
print(f"G:             {cfg.rollouts_per_q}  groups/step={cfg.dapo_groups_per_step}  rollout_batch={cfg.dapo_rollout_batch}")
print(f"LoRA:          rank={cfg.lora_rank} alpha={cfg.lora_alpha} target_all_linear={args.lora_target_all_linear}")
print(f"KL:            {cfg.kl_coef}")
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

print(f"Train: {len(base_q)} unique x {cfg.epochs} epochs = {len(train_q)} examples. Test: {len(test_q)} questions.")

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
            "lora_target_all_linear": args.lora_target_all_linear,
            "rollouts_per_q": cfg.rollouts_per_q,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "temperature": cfg.temperature,
            "max_cot_tokens": cfg.max_cot_tokens,
            "kl_coef": cfg.kl_coef,
            "correct_reward": cfg.correct_reward,
            "format_reward": cfg.format_reward,
            "train_microbatch_size": cfg.train_microbatch_size,
            "dapo_groups_per_step": cfg.dapo_groups_per_step,
            "dapo_rollout_batch": cfg.dapo_rollout_batch,
            "dapo_max_resamples": cfg.dapo_max_resamples,
            "sc_K": args.sc_K,
            "sc_temperature": args.sc_temperature,
        },
    )

print(f"Loading base model: {cfg.base_model}")
model = AutoModelForCausalLM.from_pretrained(
    cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

if cfg.use_lora:
    target_modules = (
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if args.lora_target_all_linear
        else ["q_proj", "v_proj"]
    )
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
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

vllm_engine = None
if args.use_vllm:
    if not VLLM_AVAILABLE:
        raise RuntimeError("--use_vllm passed but `vllm` is not importable. `pip install vllm`.")
    print(f"Initializing vLLM engine (gpu_memory_utilization={args.vllm_gpu_memory_utilization}, max_lora_rank={cfg.lora_rank})")
    vllm_engine = VLLMRollout(
        base_model=cfg.base_model,
        tokenizer=tokenizer,
        max_lora_rank=cfg.lora_rank,
        max_cot_tokens=cfg.max_cot_tokens,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        dtype="bfloat16",
    )
    if cfg.use_lora:
        vllm_engine.sync_lora_from_peft(model)

if args.mode in ("train", "train_eval"):
    print(f"\n=== DAPO TRAIN: {len(base_q)} questions x {cfg.epochs} epochs = {len(train_q)} examples (bucket={args.bucket}) ===")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    train_dapo(agent, train_q, train_gt, cfg, optimizer, vllm_engine=vllm_engine)

    # Make sure the final adapter is in vLLM before we move to eval
    if vllm_engine is not None and cfg.use_lora:
        vllm_engine.sync_lora_from_peft(model)

if args.mode in ("eval", "train_eval"):
    print(f"\n=== SELF-CONSISTENCY EVAL: {len(test_q)} questions, K={args.sc_K}, T={args.sc_temperature} ===")
    eval_path = os.path.join(cfg.output_dir, f"eval_sc_K{args.sc_K}.json")
    results = evaluate_sc(
        agent, test_q, test_gt, cfg,
        K=args.sc_K,
        temperature=args.sc_temperature,
        output_path=eval_path,
        batch_size=args.eval_batch_size,
        vllm_engine=vllm_engine,
    )
    print(f"\nMajority-vote accuracy: {results['accuracy_majority']:.2%} ({results['correct_majority']}/{results['total']})")
    print(f"Pass@K (any correct):   {results['accuracy_any']:.2%} ({results['correct_any']}/{results['total']})")
    print(f"Mean agreement:         {results['mean_agreement']:.2%}")

    if wandb.run is not None:
        wandb.log({
            "eval/accuracy_majority": results["accuracy_majority"],
            "eval/accuracy_pass_at_K": results["accuracy_any"],
            "eval/mean_agreement": results["mean_agreement"],
            "eval/K": args.sc_K,
            "eval/temperature": args.sc_temperature,
        })

if vllm_engine is not None:
    vllm_engine.cleanup()

if wandb.run is not None:
    wandb.finish()
