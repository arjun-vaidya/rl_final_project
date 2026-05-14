# router_solver (V1)

Hierarchical Router-Solver with code-generating Solver. Result: ~1.7% accuracy on GSM8K. Kept for reference; see `router_solver_v2/` for V2 redesign.

## Setup

- Router: emits numbered subgoals
- Solver: generates Python code per subgoal
- Execution: sandboxed Python tool
- Reward: +1 if final answer correct, +0.5 format bonus, 0 otherwise
- Training: GRPO with two LoRA adapters on shared base

## Key limitation

Outcome-only RL on multi-step pipeline: wrong final answer propagates negative gradient through all steps, even correct ones. Outcome reward + heuristic shaping insufficient signal at 1.5B with rank-8 LoRA.

## Running

```bash
python src/training/train_router_solver.py
# Or use convenience scripts: run_smoke.sh, run_slim_benchmark.sh, etc.
```

See `docs/` for design docs explaining the problem and approach.

## Special considerations

- Tool execution via multiprocessing pool
- Batched rollout optimization improves throughput 4x (but not accuracy)
- vLLM optional for faster generation
