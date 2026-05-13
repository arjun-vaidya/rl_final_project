#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot
OUT="experiments/router_objective_credit_answer_bearing_20260513_090221"
python3 -u main.py \
  --mode train \
  --checkpoint experiments/slim_g6_20260508_153045/phase4_final.pt \
  --dataset slim \
  --execution-branch soft \
  --train-questions 10 \
  --rollouts-per-q 5 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --router-temperature 0.2 \
  --solver-temperature 0.7 \
  --router-max-tokens 300 \
  --solver-max-tokens 512 \
  --train-solver-max-tokens 256 \
  --use-answer-synthesis off \
  --plan-parse-repair on \
  --strict-answer-format off \
  --router-prompt-hardening off \
  --candidate-rerank off \
  --use-judge off \
  --gradient-checkpointing on \
  --outcome-credit-mode answer_bearing \
  --save-rollout-traces on \
  --output-dir "$OUT/train" \
  --wandb-run-name "router_objective_answer_bearing_$(basename "$OUT")" 2>&1 | tee "$OUT/train.log"
python3 -u main.py \
  --mode trace_rollouts \
  --checkpoint "$OUT/train/phase4_final.pt" \
  --dataset slim \
  --execution-branch soft \
  --diagnostic-questions 10 \
  --diagnostic-rollouts-per-q 5 \
  --router-temperature 0.2 \
  --solver-temperature 0.7 \
  --router-max-tokens 300 \
  --solver-max-tokens 256 \
  --use-answer-synthesis off \
  --plan-parse-repair on \
  --strict-answer-format off \
  --router-prompt-hardening off \
  --candidate-rerank off \
  --output-dir "$OUT/eval" \
  --wandb-run-name "router_objective_answer_bearing_eval_$(basename "$OUT")" 2>&1 | tee "$OUT/eval.log"
python3 -u main.py \
  --mode taxonomy \
  --trace-input "$OUT/eval/rollout_traces.jsonl" \
  --taxonomy-output "$OUT/eval/taxonomy_report.md" 2>&1 | tee "$OUT/taxonomy.log"
