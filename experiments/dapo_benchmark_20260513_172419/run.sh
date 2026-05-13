#!/usr/bin/env bash
set -euo pipefail
set -a
source /home/machina/pvd2112/rl_final_project/.env
set +a
export PYTHONPATH=/home/machina/pvd2112/rl_final_project/linear_reasoning:/home/machina/pvd2112/rl_final_project/dapo_linear_math:
cd /home/machina/pvd2112/rl_final_project/dapo_linear_math
python3 main.py   --mode train_eval   --base_model Qwen/Qwen2.5-Math-1.5B-Instruct   --bucket mixed_hard   --rollouts_per_q 5   --dapo_groups_per_step 1   --dapo_rollout_batch 1   --dapo_max_resamples 8   --train_microbatch_size 1   --epochs 1   --learning_rate 1e-5   --lora_rank 32   --kl_coef 0.015   --sc_K 5   --sc_temperature 0.6   --eval_questions 200   --output_dir "/home/machina/pvd2112/rl_final_project/experiments/dapo_benchmark_20260513_172419/model"   --wandb_run_name dapo_benchmark_mixedhard_g5_20260513_172419
