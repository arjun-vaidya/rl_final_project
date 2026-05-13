# router_solver (V1)

The first hierarchical Router-Solver experiment, with a code-generating Solver and outcome-only rewards. Reached approximately 1.7% accuracy on GSM8K. Kept in the repo as the V1 reference point that motivated the V2 redesign in `../router_solver_v2/`.

## What this directory does

Two LoRA adapters on one Qwen2.5-1.5B-Instruct base. The Router emits a numbered subgoal list, the Solver writes a Python snippet per subgoal, and a sandboxed Python tool executes it. The outcome reward is binary on the final numeric answer; the only intermediate signal is a heuristic format and tool-call-success bonus. Training uses GRPO over `B * G` trajectories per step, sharing one optimizer between the two adapters.

## Why this design and why it failed

The motivating problem (documented in `docs/01_problem.md`): outcome-only RL on a multi-step tool-using pipeline creates gradient conflict. A wrong final answer back-propagates negative gradient through every Router subgoal and every Solver snippet, regardless of whether each individual step was correct. The hypothesis was that splitting the policy into Router and Solver heads with separate (heuristic) reward shaping would isolate the credit signal.

The result was that at 1.5B with rank-8 LoRA, the joint policy did not have either the capacity or the signal-to-noise ratio to learn the Router-Solver-tool contract from outcome reward alone. Most of the wall-clock budget went into infrastructure (vLLM memory pressure, batched-rollout rewrite, gradient checkpointing, an OOM-fallback chunked backward path) which produced a ~4x throughput improvement (235 s/step to 58 s/step at B=2, G=4) but no movement on test accuracy. Section 5.1 of the report has the full reading.

V2 (`../router_solver_v2/`) replaces the Python-tool contract with a text-reasoning Solver and adds dense judge-based rewards. The linear-policy work (`../linear_reasoning/`) drops the hierarchy entirely.

## How to run

The V1 training entrypoint is `src/training/train_router_solver.py`. The convenience scripts in the root are smoke / parity / benchmark wrappers around it:

```bash
cd router_solver

# Quick smoke run with the optimized batched-rollout path.
bash run_smoke.sh

# Run the slim-dataset benchmark (matches the dataset spec in slim_dataset_provenance).
bash run_slim_benchmark.sh

# Reference full-pass training (the configuration documented in the report).
bash run_b2_benchmark.sh

# Parity verification between the optimized batched path and the original sequential path.
bash run_parity_check.sh
```

The full-pass configuration in the reference run was `B=120 G=2 max_steps=35` with `ROUTER_SOLVER_GEN_BATCH_SIZE=32` and `ROUTER_SOLVER_LOSS_CHUNK_SIZE=4`. Run-time environment knobs are documented in `docs/OPTIMIZATION_BEST_PRACTICES.md`.

## Important files

| Path | Purpose |
|---|---|
| `src/training/train_router_solver.py` | the V1 GRPO training loop with the batched-rollout overhaul |
| `src/agents/router_solver_agent.py` | Router and Solver LoRA agents, `batched_rollout` |
| `src/env/python_tool.py` | sandboxed Python execution via a persistent multiprocessing pool |
| `docs/01_problem.md` ... `docs/08_training_pipeline_chart.md` | numbered design docs (problem statement, approach, dataset, design, evaluation, references, plan-memory extension, pipeline chart) |
| `docs/OPTIMIZATION_BEST_PRACTICES.md` | the env-var knobs (`ROUTER_SOLVER_*`) that control batching, chunking, vLLM, parity verification |
| `docs/optimization_router_solver_tagged.md` | the per-step diagnostic spec that the training loop logs against |
| `configs/` | training configs, smoke / mini / slim / benchmark variants |
| `logs/` | run logs from the reference experiments referenced in the report |
| `experiments/` | reference checkpoints and traces (the `router_solver_decomposed/versioned/final_fullpass_B120_G2_steps35` directory is the canonical run) |

## Notes and caveats

- vLLM in the V1 training loop is unstable on small GPUs (the reference experiments ran with `ROUTER_SOLVER_USE_VLLM=0`). The optimized HF path is what the recorded numbers were collected on.
- `python_tool.py` uses a persistent `multiprocessing.Pool(64)` to avoid per-call subprocess spawn overhead. If you raise the pool size, watch for file-descriptor exhaustion on long runs.
- The reward is binary on the final numeric answer. The heuristic intermediate-step bonuses are tiny and intentionally so; they were not enough signal.
- The Python sandbox is a process-level isolation only, not a security sandbox. Do not run this on untrusted prompts.
- See `docs/07_plan_memory.md` for the cross-problem plan-retrieval extension that was scoped but never implemented in V1.
