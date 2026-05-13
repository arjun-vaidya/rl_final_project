#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/experiments/slim_g6_20260508_153045/phase4_final.pt}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/parser_split_10q}"
TRACE_QUESTIONS="${TRACE_QUESTIONS:-10}"
TRACE_ROLLOUTS="${TRACE_ROLLOUTS:-6}"

mkdir -p "$OUT_DIR"

run_variant() {
  local variant="$1"
  shift
  local variant_dir="$OUT_DIR/$variant"
  mkdir -p "$variant_dir"

  PYTHONPATH=. python3 -u main.py \
    --mode trace_rollouts \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset slim \
    --diagnostic-questions "$TRACE_QUESTIONS" \
    --diagnostic-rollouts-per-q "$TRACE_ROLLOUTS" \
    --router-temperature 0.2 \
    --solver-temperature 1.0 \
    --router-max-tokens 300 \
    --solver-max-tokens 200 \
    --output-dir "$variant_dir" \
    --trace-input "$variant_dir/rollout_traces.jsonl" \
    "$@" 2>&1 | tee "$variant_dir/trace_rollouts.log"

  PYTHONPATH=. python3 -u main.py \
    --mode taxonomy \
    --trace-input "$variant_dir/rollout_traces.jsonl" \
    --taxonomy-output "$variant_dir/taxonomy_report.md" \
    --taxonomy-max-failures 500 2>&1 | tee "$variant_dir/taxonomy.log"
}

run_variant baseline
run_variant repair_fallback_only --plan-parse-repair on
run_variant router_prompt_hardening_only --router-prompt-hardening on
run_variant hardening_plus_repair --router-prompt-hardening on --plan-parse-repair on
