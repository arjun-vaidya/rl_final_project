"""Joint GRPO training for Router and Solver LoRAs (Conditions #3, #4, #4a/b/c).

Spec: docs/04_design.md §"Training loop".
Joint update: (router_loss + solver_loss).backward(). Fallback: alternating updates.
Plan Memory: pass `--memory_mode {none,random,similarity}` to switch conditions.

Usage:
    python -m src.training.train_router_solver --config configs/router_solver.yaml
    python -m src.training.train_router_solver --config configs/router_solver_memory_sim.yaml
"""


def main() -> None:
    """Load config, build RouterSolverAgent (optionally with memory), run joint GRPO."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
