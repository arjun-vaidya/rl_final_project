# Router-Solver Optimization Commit Summary (`8c5eafc`)

## Commit scope
- Commit: `8c5eafc`
- Date: 2026-05-04 23:34 UTC (local logs show this run family start time on 2026-05-05)
- Changed-file inventory captured via `git show --stat --oneline 8c5eafc` and `git show --name-only 8c5eafc`.

This commit formalized the “batched rollout + batched scoring + parity” rewrite of `train_router_solver.py`, added benchmarking/diagnostic scripts, and captured baseline/optimized run artifacts.

## Files changed in the commit
- Core training:
  - [router_solver/src/training/train_router_solver.py](router_solver/src/training/train_router_solver.py)
  - [router_solver/src/agents/router_solver_agent.py](router_solver/src/agents/router_solver_agent.py)
  - [router_solver/src/env/python_tool.py](router_solver/src/env/python_tool.py)
- Config/scripts:
  - [router_solver/requirements.txt](router_solver/requirements.txt)
  - [router_solver/run_smoke.sh](router_solver/run_smoke.sh)
  - [router_solver/run_b2_benchmark.sh](router_solver/run_b2_benchmark.sh)
  - [router_solver/run_mini_benchmark.sh](router_solver/run_mini_benchmark.sh)
  - [router_solver/run_slim_benchmark.sh](router_solver/run_slim_benchmark.sh)
  - [router_solver/run_profiler.sh](router_solver/run_profiler.sh)
  - [router_solver/run_parity_check.sh](router_solver/run_parity_check.sh)
  - [router_solver/benchmark_parity.sh](router_solver/benchmark_parity.sh)
  - [router_solver/fast_benchmark.py](router_solver/fast_benchmark.py)
- Docs/plans:
  - [router_solver/docs/OPTIMIZATION_BEST_PRACTICES.md](router_solver/docs/OPTIMIZATION_BEST_PRACTICES.md)
  - [router_solver/docs/optimization_router_solver_tagged.md](router_solver/docs/optimization_router_solver_tagged.md)
  - [router_solver/docs/plans/router_train_optimization_plan.md](router_solver/docs/plans/router_train_optimization_plan.md)
- Logs and monitoring artifacts:
  - [router_solver/*.log](router_solver)
  - [router_solver/logs/*](router_solver/logs)

## What was implemented

### 1) Training-time batching and rollout overhaul
- `agent.batched_rollout(...)` added in [router_solver/src/agents/router_solver_agent.py](router_solver/src/agents/router_solver_agent.py):
  - router plans generated in one batched call.
  - solver steps advanced in layer-wise batches.
  - tool calls executed via shared process pool in parallel.
- `train_router_solver.py` now uses `batched_rollout` for all `B*G` trajectories in a step:
  - all rollout collection occurs with `torch.inference_mode()`.
  - `gen_batch_size` control added via `ROUTER_SOLVER_GEN_BATCH_SIZE`.

### 2) Batched log-prob path and scoring efficiency
- Added batched versions:
  - `batched_teacher_forced_logprobs`
  - `batched_reference_logprobs`
- Reference-pass scoring now avoids per-sample dictionary/cached materialization patterns and supports adapter state transitions.
- Adapter/Mode safety:
  - saves/restores LoRA/model eval mode around reference scoring.

### 3) Loss robustness / OOM fallback / parity validation
- Added optional loss chunking + chunked backward:
  - `ROUTER_SOLVER_LOSS_CHUNK_SIZE` controls explicit chunking.
  - if unset, defaults to full-batch backward.
  - on OOM, auto-fallback to chunked path.
- Added parity checks with `ROUTER_SOLVER_PARITY_VERIFY`:
  - computes `live_data_gap` between full-batch and chunked objective on live rollout data.
  - logged by `tqdm.write` for monitoring.
- Added per-step visibility: gradient checkpointing, step start, rollout collect time, chunk info, runtime, ETA, and `tqdm` progress metrics.

### 4) CPU tool execution optimization
- `python_tool.py` switched from per-call subprocess spawn to a persistent `multiprocessing.Pool(64)`.
- Tool calls run via `run_python(code, timeout=...)` with timeout recovery.
- Added AST post-processing to print last expression and limit output.

### 5) Logging + experiment workflow
- Added richer logs and run scripts for apples-to-apples comparisons:
  - smoke, slim, mini, benchmark, and parity workflows.
- Created parity scripts and logs to compare sequential/optimized and vLLM/non-vLLM paths.

## Measured results from commit artifacts

### Short-step baseline vs optimized behavior
These are in root logs:
- **baseline (B=2,G=4, vLLM off):** [router_solver/baseline.log](router_solver/baseline.log)
  - `collected_rollouts n_records=8`
  - `step_time_sec=234.80`
- **optimized (B=2,G=4, vLLM off):** [router_solver/optimized.log](router_solver/optimized.log)
  - `collected_rollouts n_records=8`
  - `step_time_sec=57.99`
- This is ~4.0x faster per training step on the same batch shape.

### Parity runs (same shape)
- **parity_baseline:** [router_solver/parity_baseline.log](router_solver/parity_baseline.log) -> `step_time_sec=148.83`
- **parity_optimized:** [router_solver/parity_optimized.log](router_solver/parity_optimized.log) -> `step_time_sec=101.58`
- **vLLM parity file:** [router_solver/vllm_parity.log](router_solver/vllm_parity.log) contains both successful and failed vLLM-path attempts; used to prove vLLM memory pressure/capacity concerns.

### Full-pass pilot/final training behavior
- **Pilot:** [router_solver/logs/full_pass_B120_G2_steps35_pilot.out](router_solver/logs/full_pass_B120_G2_steps35_pilot.out) — step0 throughput estimate: `1.87h` for 35 steps.
- **Pilot:** [router_solver/logs/full_pass_B120_G2_steps38_pilot.out](router_solver/logs/full_pass_B120_G2_steps38_pilot.out) — step0 throughput estimate: `~2.06h`.
- **Pilot:** [router_solver/logs/full_pass_B120_G2_steps54_ckpt.out](router_solver/logs/full_pass_B120_G2_steps54_ckpt.out) — step0 throughput estimate: `2.77h`.
- **Pilot:** [router_solver/logs/full_pass_B120_G2_steps63_ckpt.out](router_solver/logs/full_pass_B120_G2_steps63_ckpt.out) — `~3.42h` (not meeting target).
- **G sensitivity:** [router_solver/logs/full_pass_B120_G3_steps63_ckpt.out](router_solver/logs/full_pass_B120_G3_steps63_ckpt.out) -> `~5.16h`.
- **Final selected path:** [router_solver/logs/final_fullpass_B120_G2_steps35.out](router_solver/logs/final_fullpass_B120_G2_steps35.out) -> completed in ~1h52m, avg step ~197s.

### Stability/fidelity checkpoints
- OOM fallback evidence:
  - [router_solver/logs/full_pass_B120_steps63_ckpt.out](router_solver/logs/full_pass_B120_steps63_ckpt.out) contains: `[train][oom] full-batch backward OOM, retrying with chunk_size=4`.
- Parity gap evidence:
  - [router_solver/logs/train_router_solver_nohup_foreground_run.log](router_solver/logs/train_router_solver_nohup_foreground_run.log)
  - [router_solver/smoke_run.log](router_solver/smoke_run.log)
  - both include `[train][parity] live_data_gap=0.0000000000`.

## Final training config used in this run family
The documented full-pass family used
- `ROUTER_SOLVER_BATCH_SIZE=120`
- `ROUTER_SOLVER_GROUP_SIZE=2`
- `ROUTER_SOLVER_GEN_BATCH_SIZE=32`
- `ROUTER_SOLVER_LOSS_CHUNK_SIZE=4`
- `ROUTER_SOLVER_GRADIENT_CHECKPOINTING=1`
- `ROUTER_SOLVER_TRAIN_COMPILE=0`
- `ROUTER_SOLVER_USE_VLLM=0` (used after vLLM memory-pressure experiments)
- `ROUTER_SOLVER_MAX_STEPS` set per test (typically 35 for target-satisfying runs)
- `ROUTER_SOLVER_SLIM_DATASET=0` for full dataset pass
- `ROUTER_SOLVER_PARITY_VERIFY` enabled during monitored runs

## Why this is the “documented outcome” from the commit
This commit intentionally chose speed first without removing objective logic:
- same two-stage reward decomposition (`router_reward + solver_step rewards`), same GRPO normalization and grouping semantics;
- deterministic rollout/eval boundaries and identical output/tokenization path;
- measurable throughput gains from batched rollout + batched forward scoring + parallel tool execution;
- explicit checkpointing/validation hooks to confirm parity on live data.

## Useful references for follow-up
- [router_solver/docs/optimization_router_solver_tagged.md](router_solver/docs/optimization_router_solver_tagged.md)
- [router_solver/docs/OPTIMIZATION_BEST_PRACTICES.md](router_solver/docs/OPTIMIZATION_BEST_PRACTICES.md)
- [router_solver/docs/plans/router_train_optimization_plan.md](router_solver/docs/plans/router_train_optimization_plan.md)
- [router_solver/run_smoke.sh](router_solver/run_smoke.sh)
