# Long-context Reasoning for LLMs

ORCSE6529 final project. Tests whether GRPO (as in DeepSeekMath, DeepSeek-R1) works on 1.5B models with LoRA on a single GPU.

## Project Structure

- `linear_reasoning/` GRPO on general base. Works (+1.9pp).
- `grpo_linear_math_version/` GRPO on math base. No gain.
- `dapo_linear_math/` DAPO variant on math base. +0.3pp (noise).
- `router_solver/` Hierarchical V1 (Router + code Solver). 1.7%.
- `router_solver_v2/` Hierarchical V2 (Router + text Solver + GPT judge). 35%.
- `router_solver_hierarchical_pivot/` V3 (easy/soft/hard branches + memory).
- `baseline/` Evaluator for three base models.
- `report/main.pdf` Full writeup.

## Quick start

Read `report/main.pdf` first.

Run experiments:
```bash
python linear_reasoning/main.py --mode train_eval
python grpo_linear_math_version/main.py --mode train_eval
python dapo_linear_math/main.py --mode train_eval
python baseline/eval_all_baselines.py
```

Each folder has a README with hyperparameters.

## Setup

Single L4 GPU (24GB). Each folder has its own `requirements.txt`. Dependencies: torch, transformers, peft, datasets, wandb (optional), vllm (optional).
