#!/bin/bash
export ROUTER_SOLVER_BATCH_SIZE=2
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_GEN_BATCH_SIZE=4
export ROUTER_SOLVER_MAX_STEPS=2
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export WANDB_MODE=disabled

echo "Starting mini benchmark with optimized batched rollout and vectorized log-probs..."
source .venv/bin/activate
time python -m src.training.train_router_solver
