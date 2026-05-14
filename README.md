# Long-context Reasoning for Agentic LLMs

Final project for ORCSE6529 (Advanced Reinforcement Learning), Columbia University, Spring 2026.

Authors: Arjun Vaidya, Peter Vail Driscoll.

Tests whether GRPO with verifiable rewards (as in DeepSeekMath, DeepSeek-R1) works on 1.5B models with LoRA on a single GPU. Results: GRPO works on general base (+1.9pp), fails on math-specialized base, DAPO shows +0.3pp on math base within noise. Hierarchical agents (Router-Solver V1/V2) fail due to interface design, not reasoning capability.

## Headline numbers (GSM8K, 1319-question test set, greedy)

| Setup | Accuracy |
|---|---|
| Qwen2.5-1.5B-Instruct (general base, no RL) | 69.7% |
| Qwen2.5-1.5B-Instruct + LoRA GRPO (`linear_reasoning/`) | 71.6% |
| Qwen2.5-Math-1.5B-Instruct (math base, no RL) | 84.8% |
| Qwen2.5-Math-1.5B-Instruct + LoRA GRPO (`grpo_linear_math_version/`) | 84.8% (zero RL gain) |
| Router-Solver V1 (hierarchical, code-gen Solver, `router_solver/`) | 1.7% |
| Router-Solver V2 (text Solver + GPT-4o-mini judge, `router_solver_v2/`) | ~35% (best fresh eval, synthesis on) |

The full writeup is in `report/main.pdf`.

## Directories

- `linear_reasoning/` — GRPO on general base (Qwen2.5-1.5B-Instruct). Works.
- `grpo_linear_math_version/` — GRPO on math-specialized base (Qwen2.5-Math-1.5B-Instruct). No improvement.
- `dapo_linear_math/` — DAPO variant on math base (dynamic group filtering, rank-32 LoRA, reduced KL).
- `router_solver/` — Hierarchical V1: Router emits subgoals, Solver generates code per subgoal.
- `router_solver_v2/` — Hierarchical V2: Solver generates text, GPT-4o-mini judge scores steps.
- `router_solver_hierarchical_pivot/` — Hierarchical V3: Easy/soft/hard branches, graph memory on hard.
- `baseline/` — Evaluation harness for all three base models.
- `report/` — LaTeX source and PDF writeup.

## Getting started

Start with `report/main.pdf` (4-6 pages). Describes all experiments and findings.

To run experiments:
- `linear_reasoning/main.py --mode train_eval` — General base, GRPO training
- `grpo_linear_math_version/main.py --mode train_eval` — Math base, GRPO training
- `dapo_linear_math/main.py --mode train_eval` — Math base, DAPO training
- `baseline/eval_all_baselines.py` — Evaluate three base models

Each folder has its own `README.md` with hyperparameter choices and special considerations.

## Environment

Single NVIDIA L4 GPU (24GB). Each folder has its own `requirements.txt`:
- `torch` (bf16), `transformers`, `peft` (LoRA), `datasets` (GSM8K), `wandb` (optional)
- `vllm` (optional, for dapo_linear_math and router_solver judge)

No shared top-level requirements.txt — folders are independent with different dependency pins.

## Running experiments

Baselines:
```bash
python baseline/eval_all_baselines.py
python baseline/compare_models.py
```

Training:
```bash
python linear_reasoning/main.py --mode train_eval
python grpo_linear_math_version/main.py --mode train_eval
python dapo_linear_math/main.py --mode train_eval
```

Each experiment logs to `train.log` and `runs/eval_*.json`. W&B project names match folder names.
