#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JUDGE_ENV_FILE="${SCRIPT_DIR}/../env/local_judge.env"
if [[ -f "${SCRIPT_DIR}/../env/local_judge.env.local" ]]; then
  JUDGE_ENV_FILE="${SCRIPT_DIR}/../env/local_judge.env.local"
fi

if [[ ! -f "${JUDGE_ENV_FILE}" ]]; then
  echo "Missing ${JUDGE_ENV_FILE}. Copy env/local_judge.env.example first." >&2
  exit 1
fi

# Export the sourced judge settings so the detached Python process inherits them.
set -a
# shellcheck disable=SC1090
source "${JUDGE_ENV_FILE}"
set +a

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiments/slim_g6_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"
PID_FILE="${OUT_DIR}/train.pid"

TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-120}"
EVAL_QUESTIONS="${EVAL_QUESTIONS:-100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

COMMAND=(
  python3 -u main.py
  --mode train
  --dataset slim
  --train-questions "${TRAIN_QUESTIONS}"
  --eval-questions "${EVAL_QUESTIONS}"
  --rollouts-per-q 6
  --epochs 1
  --learning-rate 1e-5
  --router-temperature 0.2
  --solver-temperature 1.0
  --router-max-tokens 300
  --solver-max-tokens 200
  --checkpoint-every 10
  --log-every 5
  --use-judge on
  --output-dir "${OUT_DIR}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  COMMAND+=(--checkpoint "${RESUME_CHECKPOINT}")
fi

(
  cd "${ROOT_DIR}"
  export PYTHONPATH=.
  exec setsid "${COMMAND[@]}"
) > "${OUT_DIR}/train.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
sleep 2

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Detached launch failed. Check ${OUT_DIR}/train.log" >&2
  exit 1
fi

echo "Launched slim dataset G=6 run."
echo "PID: ${PID}"
echo "Output directory: ${OUT_DIR}"
echo "PID file: ${PID_FILE}"
echo "Log tail command: tail -f ${OUT_DIR}/train.log"
