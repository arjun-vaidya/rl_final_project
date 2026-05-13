# Router Solver Hierarchical Pivot Workspace

This directory is a clean pivot workspace.

- `linear_base/` is a copy of the original `linear_reasoning` project.
- Router stack (for the current hierarchical pivot) is at repository root:
  - `main.py`
  - `src/`
  - `ablation_suite/`
  - `scripts/`
  - `judge_ops/`

Baseline comparison artifacts:
- `experiments/constrained_final_decode_10q/sota_baseline/` (baseline run with 26.67% @10Q, G=6)
- `experiments/phase4_final.pt` symlink to canonical checkpoint used by baseline scripts.

Primary evaluation contract (MUST remain fixed):
- Diagnostics on first 10 questions
- `--diagnostic-rollouts-per-q 6` (minimum G=6)
- Extrapolate to 50Q by `first_10_result * 5`
