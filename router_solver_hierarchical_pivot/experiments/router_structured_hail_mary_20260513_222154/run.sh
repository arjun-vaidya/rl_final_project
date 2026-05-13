#!/usr/bin/env bash
set -euo pipefail
cd /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot
outdir="/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_structured_hail_mary_20260513_222154"
while ps -p 405398 >/dev/null 2>&1; do
  sleep 20
done
python3 main.py   --mode train   --checkpoint experiments/slim_g6_20260508_153045/phase4_final.pt   --dataset slim   --execution-branch soft   --train-questions 10   --rollouts-per-q 5   --epochs 1   --learning-rate 1e-5   --router-temperature 0.2   --solver-temperature 0.7   --router-max-tokens 300   --solver-max-tokens 512   --train-solver-max-tokens 256   --use-answer-synthesis off   --plan-parse-repair on   --strict-answer-format off   --router-prompt-hardening off   --candidate-rerank off   --use-judge off   --gradient-checkpointing on   --informative-group-sampling on   --informative-group-mode structured   --informative-max-resamples 8   --outcome-credit-mode structured_component   --output-dir "$outdir/train"   --wandb-run-name router_structured_hail_mary_train_20260513_222154
python3 main.py   --mode trace_rollouts   --checkpoint "$outdir/train/phase4_final.pt"   --dataset slim   --execution-branch soft   --eval-questions 10   --rollouts-per-q 5   --router-temperature 0.2   --solver-temperature 0.7   --router-max-tokens 300   --solver-max-tokens 256   --use-answer-synthesis off   --plan-parse-repair on   --strict-answer-format off   --router-prompt-hardening off   --candidate-rerank off   --output-dir "$outdir/eval"   --wandb-run-name router_structured_hail_mary_eval_20260513_222154
python3 main.py   --mode taxonomy   --trace-input "$outdir/eval/rollout_traces.jsonl"   --diagnostic-output "$outdir/eval/taxonomy_report.md"   --no-wandb
