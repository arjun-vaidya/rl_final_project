"""GRPO training for the flat agent (Condition #2).

Spec: docs/04_design.md §"GRPO hyperparameters" and §"Training loop".
Uses trl.GRPOTrainer or equivalent. Outcome-only reward.

Usage:
    python -m src.training.train_flat --config configs/flat.yaml
"""


def main() -> None:
    """Load config, build FlatAgent + reward fn, run GRPO, log to wandb."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
