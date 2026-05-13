# Long-context Reasoning for Agentic LLMs

Final project for ORCSE6529 (Advanced Reinforcement Learning), Columbia University, Spring 2026.

Authors: Arjun Vaidya, Peter Vail Driscoll.

The project studies whether reinforcement learning recipes that work at frontier scale (GRPO with verifiable rewards, as in DeepSeekMath and DeepSeek-R1) transfer to small 1.5B language models trained with LoRA on a single GPU. The headline result is a sequence of negative findings that each isolate a specific failure mode, plus one positive baseline. We then design a recipe (in `dapo_linear_math/`) that targets the failure mode of the math-base run directly.

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

## What's in this repo

```
linear_reasoning/             linear CoT policy with GRPO on the general Qwen base (works, +1.9 pts)
grpo_linear_math_version/     same recipe on the math-specialized base (no movement, diagnosed)
dapo_linear_math/             DAPO-style follow-up to the math-base run (dynamic sampling, bigger LoRA, vLLM-optional)
router_solver/                hierarchical Router-Solver V1, code-generating Solver
router_solver_v2/             hierarchical V2, text Solver + GPT-4o-mini judge for dense rewards
router_solver_hierarchical_pivot/   ongoing pivot: easy/soft/hard branches + Hopfield-style memory retrieval
baseline/                     three-model GSM8K eval harness (general, math, RL'd)
data_probing/                 per-question difficulty probe on GSM8K-train (drives the partition.json buckets)
report/                       LaTeX source + compiled PDF for the final report
issues/                       eight design-doc writeups of architectural problems we hit and what we did about them
notes/                        date-stamped status updates and earlier plans (research record, not docs)
project_description           the course-assigned problem statement
```

## How to read this repo

The story flows in roughly this order:

1. **Start in `report/main.pdf`.** It's the 4-6 page writeup. Six sections: intro, background, linear policy on the general base, linear policy on the math base, hierarchical Router-Solver V1/V2, benchmarks.
2. **`linear_reasoning/`** is the core training scaffolding. The agent, reward, GRPO loop, and eval all live here. Every other `*_linear_*` folder reuses these modules via `sys.path`.
3. **`data_probing/`** explains why we partition GSM8K into trivial / mixed / hard buckets. The probe shows that on Qwen2.5-Math-1.5B-Instruct, 88% of GSM8K-train is already solved on every rollout (zero advantage), 4% is unsolved on every rollout (also zero advantage), and only 8% is in the band where GRPO can actually learn.
4. **`grpo_linear_math_version/`** is the experiment that produced the central negative result. The math base is strong enough that 70% of training groups are uninformative under vanilla GRPO, and the KL pinned at 0.04 within five steps.
5. **`dapo_linear_math/`** is the direct response: DAPO-style dynamic sampling (drop groups with no within-group variance and refill), rank-32 LoRA on all linear modules, KL dropped to 0.015, optional vLLM rollouts, self-consistency K=8 at eval.
6. **`router_solver/` and `router_solver_v2/`** are the hierarchical experiments. V1's failure mode is documented in `issues/04_hierarchical_architecture_failures.md`. V2's failure taxonomy (51.7% of failures are architectural rather than reasoning) is in `notes/2026-05-09_peter_pov_answer_target_vs_core_reasoning.md`.
7. **`router_solver_hierarchical_pivot/`** is the ongoing work that comes after V2. Three branches (easy/soft/hard), the hard branch attaches a graph-retrieval memory module.

## Environment

We run on a single NVIDIA L4 GPU (24GB) on a GCP VM, plus a separate VM for the GPT-4o-mini judge in V2.

Each Python folder has its own `requirements.txt`. The training code expects:
- `torch` (bf16 capable),
- `transformers`,
- `peft` (LoRA),
- `datasets` (for GSM8K),
- `wandb` (logging, optional with `--no_wandb`),
- `vllm` (optional, used by `dapo_linear_math` and the V2 judge VM).

There is no top-level `requirements.txt` because the folders are independent runs with different dep pins.

## Reproducing the headline table

```bash
# Three baselines on the full 1319-question test set.
python baseline/eval_all_baselines.py
python baseline/compare_models.py
# Produces baseline/baseline_results_all.json and baseline/comparison_report.md.
```

The two RL'd numbers come from running `linear_reasoning/main.py` and `grpo_linear_math_version/main.py` end-to-end. The Router-Solver numbers come from running each subdirectory's `main.py` with the configs documented in their READMEs.

## Conventions

- All experiments emit a `train.log` and a `runs/eval_*.json` next to the script that produced them.
- W&B project names are folder-named (`grpo_linear_math`, `dapo_linear_math`, `linear_reasoning`, etc.).
- LoRA checkpoints save the full model state dict by default; this is wasteful (the base is frozen) but makes the checkpoints self-contained.
- We use `\boxed{N}` as the answer contract. Reward is +1.0 for a correct boxed answer, +0.5 format bonus for any parseable boxed token, 0 otherwise.
