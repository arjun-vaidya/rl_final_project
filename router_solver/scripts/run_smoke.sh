#!/bin/bash
# SMOKE RUN — Target ~20 minutes
# B=16 (32 rollouts/step with G=2), shorter per-question rollout chain
# expected materially lower than previous 4×-subgoal depth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_MODE=disabled

export ROUTER_SOLVER_BATCH_SIZE="${ROUTER_SOLVER_BATCH_SIZE:-16}"
export ROUTER_SOLVER_GROUP_SIZE="${ROUTER_SOLVER_GROUP_SIZE:-2}"
export ROUTER_SOLVER_GEN_BATCH_SIZE="${ROUTER_SOLVER_GEN_BATCH_SIZE:-32}"
export ROUTER_SOLVER_LOSS_CHUNK_SIZE="${ROUTER_SOLVER_LOSS_CHUNK_SIZE:-4}"
export ROUTER_SOLVER_TRAIN_COMPILE="${ROUTER_SOLVER_TRAIN_COMPILE:-0}"
export ROUTER_SOLVER_SLIM_DATASET="${ROUTER_SOLVER_SLIM_DATASET:-1}"
export ROUTER_SOLVER_USE_VLLM="${ROUTER_SOLVER_USE_VLLM:-0}"
export ROUTER_SOLVER_MAX_STEPS="${ROUTER_SOLVER_MAX_STEPS:-10}"
export ROUTER_SOLVER_PARITY_VERIFY="${ROUTER_SOLVER_PARITY_VERIFY:-1}"
export TOKENIZERS_PARALLELISM=false

source "$SCRIPT_DIR/.venv/bin/activate"
echo "=== SMOKE RUN: ${ROUTER_SOLVER_MAX_STEPS:-10} steps, B=${ROUTER_SOLVER_BATCH_SIZE} G=${ROUTER_SOLVER_GROUP_SIZE}, gen_batch=${ROUTER_SOLVER_GEN_BATCH_SIZE} ==="
echo "Start: $(date)"
python -m src.training.train_router_solver 2>&1 | tee smoke_run.log
echo "End: $(date)"
