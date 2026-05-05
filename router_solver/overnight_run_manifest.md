# Overnight Run Manifest (Finalized Run)

## Run identity
- Manifest ID: `final_fullpass_B120_G2_steps35`
- Log: [logs/final_fullpass_B120_G2_steps35.out](/home/pvd2112/rl_final_project/router_solver/logs/final_fullpass_B120_G2_steps35.out)
- Log alias: [logs/final_fullpass_B120_G2_steps35.log](/home/pvd2112/rl_final_project/router_solver/logs/final_fullpass_B120_G2_steps35.log) *(symlink to the same file)*
- W&B run local: `router_solver/wandb/run-20260505_080608-j7h3rnpx`
- W&B remote: `https://wandb.ai/peterdriscoll27-columbia-university/router-solver/runs/j7h3rnpx`
- Start/finish evidence:
  - File timestamp: `2026-05-05 11:10:12 UTC` (file mtime)
  - WandB run prefix in log: `run-20260505_080608-*` (suggests start near 08:06 UTC)
  - Total elapsed wall time: `1:55:42` from tqdm output

## Inputs / configuration
- Code baseline: merged `main` branch state at the time of the run (`main` commit `5630ab59fb04da7adeb60a2b323fc69758a9ad2d`)
- Dataset: full dataset mode (non-slim; log does not print slim marker)
- Training mode: decomposed reward (`reward_mode=decomposed`)
- Core hyperparameters captured in log:
  - `B=120`
  - `G=2`
  - `beta=0.04`
  - `max_steps=35`
  - `total_records=240` per step (`B × G`)
- Memory / stability settings observed:
  - `gradient checkpointing enabled`
  - `loss chunking requested: chunk_size=4`
  - `chunked backward will use 60 chunk(s)`

## Key outcomes
- Final completion: `[train] done. saved experiments/router_solver_decomposed/final_hierarchical_model`
- Steps completed: `35`
- Avg step time trend stabilized around `~198s`/step
- Max resident memory observed: `~8.18 GB`
- Final step metrics:
  - `loss=-3.1875`
  - `outcome_acc=0.00833`
  - `router_r=0.165`
  - `solver_r=0.000`
  - `invalid_plans=142`

## Ancillary checks / artifacts
- Parity smoke run used for correctness spot-check:
  - [logs/train_router_solver_nohup_foreground_run.log](/home/pvd2112/rl_final_project/router_solver/logs/train_router_solver_nohup_foreground_run.log)
  - includes `[train][parity] live_data_gap=0.0000000000`
- Prior pilot/full sweep artifacts:
  - [logs/full_pass_B120_G2_steps35_pilot.out](/home/pvd2112/rl_final_project/router_solver/logs/full_pass_B120_G2_steps35_pilot.out)
  - [logs/full_pass_B120_G2_steps54_ckpt.out](/home/pvd2112/rl_final_project/router_solver/logs/full_pass_B120_G2_steps54_ckpt.out)
  - [logs/full_pass_B120_G2_steps63_ckpt.out](/home/pvd2112/rl_final_project/router_solver/logs/full_pass_B120_G2_steps63_ckpt.out)
  - [logs/full_pass_B120_G3_steps63_ckpt.out](/home/pvd2112/rl_final_project/router_solver/logs/full_pass_B120_G3_steps63_ckpt.out)

## Run note
- This is a production-style overnight-style pass log for the final `B=120, G=2, 35` configuration that met the 3h target with strong progress and completed successfully.
