#!/usr/bin/env bash
set -euo pipefail
cd /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot
OUT="experiments/router_objective_credit_o2_nojudge_20260513_055720"
for VARIANT in baseline all_steps; do
  EXTRA=()
  if [ "$VARIANT" = all_steps ]; then EXTRA=(--outcome-credit-all-steps on); fi
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
    --use-answer-synthesis off \
    --plan-parse-repair on \
    --strict-answer-format off \
    --router-prompt-hardening off \
    --candidate-rerank off \
    --use-judge off \
    --save-rollout-traces on \
    --output-dir "$OUT/o2_${VARIANT}/train" \
    --wandb-run-name "router_objective_o2_nojudge_${VARIANT}_$(basename "$OUT")" \
    "${EXTRA[@]}" 2>&1 | tee "$OUT/o2_${VARIANT}_train.log"
  python3 -u main.py \
    --mode trace_rollouts \
    --checkpoint "$OUT/o2_${VARIANT}/train/phase4_final.pt" \
    --dataset slim \
    --execution-branch soft \
    --diagnostic-questions 10 \
    --diagnostic-rollouts-per-q 5 \
    --router-temperature 0.2 \
    --solver-temperature 0.7 \
    --router-max-tokens 300 \
    --solver-max-tokens 512 \
    --use-answer-synthesis off \
    --plan-parse-repair on \
    --strict-answer-format off \
    --router-prompt-hardening off \
    --candidate-rerank off \
    --output-dir "$OUT/o2_${VARIANT}/eval" \
    --wandb-run-name "router_objective_o2_nojudge_${VARIANT}_eval_$(basename "$OUT")" 2>&1 | tee "$OUT/o2_${VARIANT}_eval.log"
  python3 -u main.py \
    --mode taxonomy \
    --trace-input "$OUT/o2_${VARIANT}/eval/rollout_traces.jsonl" \
    --taxonomy-output "$OUT/o2_${VARIANT}/eval/taxonomy_report.md" 2>&1 | tee "$OUT/o2_${VARIANT}_taxonomy.log"
done
