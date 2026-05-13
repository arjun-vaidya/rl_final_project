#!/usr/bin/env bash
set -euo pipefail
set -a
source /home/machina/pvd2112/rl_final_project/.env
set +a
cd /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot
python3 main.py   --mode trace_rollouts   --checkpoint /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_dapo_port_20260513_172645/informative_all_steps/train/phase4_final.pt   --dataset slim   --execution-branch soft   --eval-questions 10   --rollouts-per-q 5   --router-temperature 0.2   --solver-temperature 0.7   --router-max-tokens 300   --solver-max-tokens 256   --use-answer-synthesis off   --plan-parse-repair on   --strict-answer-format off   --router-prompt-hardening off   --candidate-rerank off   --output-dir "/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_dapo_min_matrix_20260513_185346/all_steps_eval_g5/eval"   --wandb-run-name router_dapo_min_all_steps_eval_g5_20260513_185346
python3 main.py   --mode taxonomy   --trace-input "/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_dapo_min_matrix_20260513_185346/all_steps_eval_g5/eval/rollout_traces.jsonl"   --taxonomy-output "/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_dapo_min_matrix_20260513_185346/all_steps_eval_g5/eval/taxonomy_report.md"
