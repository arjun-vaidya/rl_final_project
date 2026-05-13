#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/machina/pvd2112/rl_final_project/router_solver_v2"
CKPT="$ROOT/experiments/slim_g6_20260508_153045/phase4_final.pt"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="$ROOT/experiments/strict_answer_format_matrix_${STAMP}"
STATUS_FILE="$BASE_OUT/matrix_status.txt"

mkdir -p "$BASE_OUT"

export PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMON_ARGS=(
  --mode trace_rollouts
  --checkpoint "$CKPT"
  --dataset slim
  --diagnostic-questions 10
  --diagnostic-rollouts-per-q 4
  --router-temperature 0.2
  --solver-temperature 1.0
  --router-max-tokens 300
  --solver-max-tokens 512
  --synthesis-max-tokens 128
  --use-answer-synthesis on
  --plan-parse-repair on
)

run_variant() {
  local name="$1"
  shift

  local out="$BASE_OUT/$name"
  local trace="$out/rollout_traces.jsonl"
  mkdir -p "$out"

  {
    echo "variant=$name"
    date -Is
    printf 'extra_args=%s\n' "$*"
  } > "$STATUS_FILE"

  echo "=== Running $name ==="
  (
    cd "$ROOT"
    python3 -u main.py \
      "${COMMON_ARGS[@]}" \
      --output-dir "$out" \
      --rollout-trace-path "$trace" \
      "$@" \
      | tee "$out/trace_rollouts.log"
  )

  (
    cd "$ROOT"
    python3 -u main.py \
      --mode taxonomy \
      --trace-input "$trace" \
      --taxonomy-output "$out/taxonomy_report.md" \
      --taxonomy-max-failures 500 \
      2>&1 | tee "$out/taxonomy.log"
  )

  echo "completed=$name" >> "$STATUS_FILE"
}

run_variant "strict_answer_format" \
  --strict-answer-format on

{
  echo "status=done"
  date -Is
} >> "$STATUS_FILE"

echo "Matrix complete: $BASE_OUT"
