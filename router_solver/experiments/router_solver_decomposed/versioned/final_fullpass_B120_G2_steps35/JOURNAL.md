# Experiment Journal: final_fullpass_B120_G2_steps35

DATE: 2026-05-05
Author: Peter Driscoll (peterdriscoll27)
Branch: main (commit: 5630ab59fb04da7adeb60a2b323fc69758a9ad2d)
WandB run: https://wandb.ai/peterdriscoll27-columbia-university/router-solver/runs/j7h3rnpx

## Run summary
- Target: Full dataset hierarchical router-solver GRPO run
- Goal: meet ~3h overnight runtime target while keeping fidelity constraints
- Run config: B=120, G=2, 35 steps, decomposed reward, chunked backward, vLLM off
- Start marker: `router_solver/logs/final_fullpass_B120_G2_steps35.out`
- End marker: `[train] done. saved experiments/router_solver_decomposed/final_hierarchical_model`

## Observed results
- Total run duration: ~1:55:42 (from tqdm summary in run log)
- Avg step time: ~197-199s/step
- Max memory: ~8.18 GB
- Final outcome:
  - loss=-3.1875
  - outcome_acc=0.00833
  - router_r=0.165
  - solver_r=0.000
  - invalid_plans=142

## Artifacts captured
- Final checkpoint package: `artifacts/final_hierarchical_model`
  - Copied from `experiments/router_solver_decomposed/final_hierarchical_model`
  - `router_lora/adapter_model.bin`
  - `router_lora/adapter_config.json`
  - `solver_lora/adapter_model.bin`
  - `solver_lora/adapter_config.json`
- Run manifest: `overnight_run_manifest.md`
- Training summary: `train_router_solver_8c5eafc_summary.md`
- Full run output log: `logs/final_fullpass_B120_G2_steps35.out`

## Benchmark and parity evidence
- Short benchmark traces:
  - [baseline] `baseline.log`
  - [optimized] `optimized.log`
  - [slim] `slim_benchmark.log`
  - [b2_benchmark] `b2_benchmark.log`
  - [mini_benchmark] `mini_benchmark.log`
  - [parity] `parity_baseline.log`, `parity_optimized.log`
  - [vLLM parity] `vllm_parity.log`
  - [smoke] `smoke_run.log`
- Parity spot-check run:
  - `train_router_solver_nohup_foreground_run.log` (live_data_gap=0.0000000000)
- Full-pass sweeps:
  - `full_pass_B120_G2_steps35_pilot.out`
  - `full_pass_B120_G2_steps54_ckpt.out`
  - `full_pass_B120_G2_steps63_ckpt.out`
  - `full_pass_B120_G3_steps63_ckpt.out`
