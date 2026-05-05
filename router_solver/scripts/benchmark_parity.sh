#!/bin/bash
export WANDB_MODE=disabled
export ROUTER_SOLVER_BATCH_SIZE=2
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_MAX_STEPS=1
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export ROUTER_SOLVER_TRAIN_COMPILE=0

echo "=== Running Baseline (Sequential, No vLLM) ==="
export ROUTER_SOLVER_USE_VLLM=0
export ROUTER_SOLVER_GEN_BATCH_SIZE=1
source .venv/bin/activate
python -m src.training.train_router_solver > baseline.log 2>&1

echo "=== Running Optimized (Batched + SDPA + Multiprocessing) ==="
export ROUTER_SOLVER_USE_VLLM=0
export ROUTER_SOLVER_GEN_BATCH_SIZE=4
python -m src.training.train_router_solver > optimized.log 2>&1

echo "--- Baseline Stats ---"
grep -E "\[train\]\[step=0\].*step_time_sec" baseline.log
echo "--- Optimized Stats ---"
grep -E "\[train\]\[step=0\].*step_time_sec" optimized.log
