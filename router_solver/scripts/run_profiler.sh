#!/bin/bash
export WANDB_MODE=disabled
export ROUTER_SOLVER_BATCH_SIZE=2
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_GEN_BATCH_SIZE=4
export ROUTER_SOLVER_MAX_STEPS=1
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export ROUTER_SOLVER_TRAIN_COMPILE=0
export ROUTER_SOLVER_PROFILE_STEPS=1
export ROUTER_SOLVER_USE_VLLM=0

source .venv/bin/activate
echo "Running profiler..."
python -m src.training.train_router_solver > profiler.log 2>&1
echo "Done. Extracted profiler table:"
grep -A 25 "profile_top_cpu_cuda=" profiler.log
