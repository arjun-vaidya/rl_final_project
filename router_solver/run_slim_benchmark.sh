#!/bin/bash
export WANDB_MODE=disabled
export ROUTER_SOLVER_BATCH_SIZE=8
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_GEN_BATCH_SIZE=32
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export ROUTER_SOLVER_TRAIN_COMPILE=0
export ROUTER_SOLVER_SLIM_DATASET=1
export ROUTER_SOLVER_USE_VLLM=1
export ROUTER_SOLVER_MAX_STEPS=2
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate
echo "=== Running Slim Benchmark: vLLM + Prefix Cache (B=8, no CPU offload) ==="
python -m src.training.train_router_solver 2>&1 | tee slim_benchmark.log
