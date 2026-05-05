#!/bin/bash
# Parity Verification: Baseline (sequential, no vLLM) vs Optimized (vLLM + batched)
# Runs 1 step with B=2, G=4 so both produce 8 rollouts from the same 2 questions.
# We compare: loss, outcome_acc, router_reward, solver_reward, step_time, max_mem_mb
set -e

export WANDB_MODE=disabled
export ROUTER_SOLVER_BATCH_SIZE=2
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_MAX_STEPS=1
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export ROUTER_SOLVER_TRAIN_COMPILE=0

source .venv/bin/activate

echo "=== PARITY CHECK: Baseline (Sequential, no vLLM) ==="
export ROUTER_SOLVER_USE_VLLM=0
export ROUTER_SOLVER_GEN_BATCH_SIZE=1
python -m src.training.train_router_solver > parity_baseline.log 2>&1
echo "Baseline done."

echo "=== PARITY CHECK: Optimized (vLLM + Batched + Prefix Cache) ==="
export ROUTER_SOLVER_USE_VLLM=1
export ROUTER_SOLVER_GEN_BATCH_SIZE=8
python -m src.training.train_router_solver > parity_optimized.log 2>&1
echo "Optimized done."

echo ""
echo "========================================="
echo "  PARITY COMPARISON"
echo "========================================="
echo ""
echo "--- Baseline ---"
grep -E "^step=" parity_baseline.log || echo "(no step line found)"
grep -E "step_time_sec" parity_baseline.log | tail -1
echo ""
echo "--- Optimized ---"
grep -E "^step=" parity_optimized.log || echo "(no step line found)"
grep -E "step_time_sec" parity_optimized.log | tail -1
echo ""
echo "========================================="
