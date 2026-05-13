#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/experiments/slim_g6_20260508_153045/phase4_final.pt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/robust_answer_suite_10q}"
TRACE_QUESTIONS="${TRACE_QUESTIONS:-10}"
TRACE_ROLLOUTS="${TRACE_ROLLOUTS:-6}"
ROUTER_MAX_TOKENS="${ROUTER_MAX_TOKENS:-300}"
SOLVER_MAX_TOKENS="${SOLVER_MAX_TOKENS:-512}"
SYNTHESIS_MAX_TOKENS="${SYNTHESIS_MAX_TOKENS:-256}"

mkdir -p "$OUT_DIR"

variant_dir="$OUT_DIR/robust_synthesis_guarded"
mkdir -p "$variant_dir"

PYTHONPATH=. python3 -u main.py \
  --mode trace_rollouts \
  --checkpoint "$CHECKPOINT_PATH" \
  --dataset slim \
  --diagnostic-questions "$TRACE_QUESTIONS" \
  --diagnostic-rollouts-per-q "$TRACE_ROLLOUTS" \
  --router-temperature 0.2 \
  --solver-temperature 1.0 \
  --router-max-tokens "$ROUTER_MAX_TOKENS" \
  --solver-max-tokens "$SOLVER_MAX_TOKENS" \
  --synthesis-max-tokens "$SYNTHESIS_MAX_TOKENS" \
  --use-answer-synthesis on \
  --constrained-final-answer-decoding on \
  --plan-parse-repair on \
  --synthesis-self-consistency on \
  --synthesis-self-consistency-samples 5 \
  --heuristic-final-selector-refined on \
  --guarded-heuristic-fallback on \
  --answer-bearing-step-hint on \
  --output-dir "$variant_dir" \
  --trace-input "$variant_dir/rollout_traces.jsonl" \
  2>&1 | tee "$variant_dir/trace_rollouts.log"

PYTHONPATH=. python3 -u main.py \
  --mode taxonomy \
  --trace-input "$variant_dir/rollout_traces.jsonl" \
  --taxonomy-output "$variant_dir/taxonomy_report.md" \
  --taxonomy-max-failures 500 \
  2>&1 | tee "$variant_dir/taxonomy.log"
