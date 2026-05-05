#!/bin/bash
# Quick B=2 benchmark to see the effect of max_model_len + parallel tools
export WANDB_MODE=disabled
export ROUTER_SOLVER_BATCH_SIZE=2
export ROUTER_SOLVER_GROUP_SIZE=4
export ROUTER_SOLVER_GEN_BATCH_SIZE=8
export ROUTER_SOLVER_LOSS_CHUNK_SIZE=4
export ROUTER_SOLVER_TRAIN_COMPILE=0
export ROUTER_SOLVER_USE_VLLM=1
export ROUTER_SOLVER_MAX_STEPS=2
export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate
echo "=== B=2 Benchmark: vLLM + max_model_len=4096 + parallel tools ==="
python -m src.training.train_router_solver 2>&1 | tee b2_benchmark.log
