# Router–Solver: Hierarchical RL for Tool-Using LLM Agents

**Problem.** Outcome-only RL on multi-step tool use creates gradient conflict — good intermediate tool calls get penalized when the final answer is wrong, bad ones get rewarded when the answer is lucky. [details →](docs/01_problem.md)

**Approach.** Split the agent into two LoRAs on one base model: a **Router** that plans subgoals, and a **Solver** that executes each subgoal with a tool call. Give each component its own reward, trained jointly with GRPO. [details →](docs/02_approach.md)

**Experiment.** Four matched-compute conditions on GSM8K + Python tool: SFT baseline, flat GRPO (outcome-only), Router–Solver (outcome-only), Router–Solver (decomposed rewards). Primary comparison: flat vs full Router–Solver. [details →](docs/05_evaluation.md)

## Docs

| # | File | Contents |
|---|---|---|
| 01 | [Problem](docs/01_problem.md) | Gradient conflict, worked example |
| 02 | [Approach](docs/02_approach.md) | Router–Solver, reward decomposition |
| 03 | [Dataset & env](docs/03_dataset.md) | GSM8K + Python tool |
| 04 | [Design](docs/04_design.md) | Model, LoRAs, GRPO, prompts, rewards |
| 05 | [Evaluation](docs/05_evaluation.md) | Conditions, metrics, ablations, success criteria |
| 06 | [References](docs/06_references.md) | Prior work + novelty statement |
| 07 | [Plan Memory](docs/07_plan_memory.md) | **Uniqueness angle** — optional cross-problem plan retrieval extension |
| 08 | [Training Pipeline Chart](docs/08_training_pipeline_chart.md) | SFT vs GRPO **data-driven** runtime + quality charts |

## Status

Docs only — no code yet. **Deadlines:** slides May 3 · talk May 4–6 · report May 15.

## Team Notes — Runtime and Optimization Summary

DATE: 2026-05-05  
Signed: Peter Driscoll (peterdriscoll27), RL Project Team

- Optimization and training run summary (post-overwrite and benchmarking): [train_router_solver_8c5eafc_summary.md](/home/pvd2112/rl_final_project/router_solver/train_router_solver_8c5eafc_summary.md)
- Overnight full-run manifest and observed final configuration: [overnight_run_manifest.md](/home/pvd2112/rl_final_project/router_solver/overnight_run_manifest.md)
- Slim benchmark dataset provenance and HF pointer: [slim_dataset_provenance.md](/home/pvd2112/rl_final_project/router_solver/slim_dataset_provenance.md)
  - Hugging Face dataset folder: https://huggingface.co/datasets/pvd232/rlfp/tree/main/rlfp_router_solver_slim_v1
  - HF commit: https://huggingface.co/datasets/pvd232/rlfp/commit/41181ad0cd1031e20a8e50cd1169d2627e4129fe
- Versioned experiment package (weights + journal + manifest + benchmark/parity references): [experiments/router_solver_decomposed/versioned/final_fullpass_B120_G2_steps35](/home/pvd2112/rl_final_project/router_solver/experiments/router_solver_decomposed/versioned/final_fullpass_B120_G2_steps35)

## Has this been done?

**Partly, yes.** The high-level idea — hierarchical planner/executor LLM agents trained with GRPO-style RL — has recent published work (Agent-as-Tool, ArCHer, AgentPRM, Tree-GRPO). Our contributions are:

1. A *clean, matched-compute comparison* isolating architecture from reward decomposition, plus a gradient-conflict diagnostic.
2. **Plan Memory** ([doc 07](docs/07_plan_memory.md)) — a cross-problem retrieval module attached to the Router, which (to our knowledge) no existing hierarchical-agent-RL paper has. This is the uniqueness angle and it's designed to be **additive** — the core experiment runs without it.

Details and citations in [docs/06_references.md](docs/06_references.md).
