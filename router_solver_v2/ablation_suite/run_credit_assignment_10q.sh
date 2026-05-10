#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$ROOT_DIR/experiments/slim_g6_20260508_153045/phase4_final.pt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/credit_assignment_10q}"
TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-10}"
TRACE_QUESTIONS="${TRACE_QUESTIONS:-10}"
TRACE_ROLLOUTS="${TRACE_ROLLOUTS:-6}"
USE_JUDGE="${USE_JUDGE:-off}"

mkdir -p "$OUT_DIR"

run_train_then_eval() {
  local variant="$1"
  shift
  local variant_dir="$OUT_DIR/$variant"
  mkdir -p "$variant_dir"

  PYTHONPATH=. python3 -u main.py \
    --mode train \
    --checkpoint "$BASE_CHECKPOINT" \
    --dataset slim \
    --train-questions "$TRAIN_QUESTIONS" \
    --rollouts-per-q 6 \
    --epochs 1 \
    --learning-rate 1e-5 \
    --router-temperature 0.2 \
    --solver-temperature 1.0 \
    --router-max-tokens 300 \
    --solver-max-tokens 200 \
    --synthesis-max-tokens 64 \
    --use-answer-synthesis on \
    --plan-parse-repair on \
    --use-judge "$USE_JUDGE" \
    --save-rollout-traces on \
    --checkpoint-every 50 \
    --log-every 2 \
    --output-dir "$variant_dir/train" \
    "$@" 2>&1 | tee "$variant_dir/train.log"

  PYTHONPATH=. python3 -u main.py \
    --mode trace_rollouts \
    --checkpoint "$variant_dir/train/phase4_final.pt" \
    --dataset slim \
    --diagnostic-questions "$TRACE_QUESTIONS" \
    --diagnostic-rollouts-per-q "$TRACE_ROLLOUTS" \
    --router-temperature 0.2 \
    --solver-temperature 1.0 \
    --router-max-tokens 300 \
    --solver-max-tokens 200 \
    --synthesis-max-tokens 64 \
    --use-answer-synthesis on \
    --plan-parse-repair on \
    --output-dir "$variant_dir/eval" \
    --trace-input "$variant_dir/eval/rollout_traces.jsonl" 2>&1 | tee "$variant_dir/eval_trace.log"

  PYTHONPATH=. python3 -u main.py \
    --mode taxonomy \
    --trace-input "$variant_dir/eval/rollout_traces.jsonl" \
    --taxonomy-output "$variant_dir/eval/taxonomy_report.md" \
    --taxonomy-max-failures 500 2>&1 | tee "$variant_dir/eval_taxonomy.log"
}

run_train_then_eval control_default_credit
run_train_then_eval variant_outcome_credit_all_steps --outcome-credit-all-steps on
