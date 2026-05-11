# Status Update (Peter) - 2026-05-10

## Scope
This update covers the router-solver-v2 experiment cycle on judge deployment, non-core inference ablations, and training-objective ablations, including the promoted 50Q outcome-heavy run.

## Infra and Runability
- Remote judge endpoint is required and now validated as working through the exported env path in `router_solver_v2/judge_ops/env/local_judge.env.local`.
- A previous failure mode (`localhost:11434` fallback) was traced to non-exported env vars after `source`; fixed by using:
  - `set -a`
  - `source .../local_judge.env.local`
  - `set +a`
- Judge path remained stable in successful runs (no recurrent endpoint failures after fix).

## Model / Checkpoint Lineage
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA adapters (`router`, `solver`).
- Prior full slim reference checkpoint:
  - `router_solver_v2/experiments/slim_g6_20260508_153045/phase4_final.pt`
- Training ablations and promoted follow-up were initialized from that checkpoint.

## Key Diagnostic Findings (Why We Plateau)
- Earlier runs plateaued in the low-20s due to sparse outcome signal and misaligned dense rewards.
- Parser/extraction fixes removed dead outcome-signal regime but did not fully solve correctness.
- Taxonomy showed mixed failures:
  - Core reasoning (`wrong_numeric_final`)
  - Non-core answer-target failures (`copied_intermediate_as_final`, `correct_number_in_trace_wrong_final`, `plan_endpoint_mismatch`)
- Outcome signal improved post-fix, but still sparse at group level.

## Inference-Side Non-Core Ablation Summary
- Large matrix and replay work was completed across synthesis/finalizer variants.
- Strongest short-slice winners were unstable across broader slices.
- Heuristic and self-consistency finalizers showed mixed tradeoffs; no durable, dominant inference-only fix emerged at scale.
- `synthesis + repair_fallback_only` remained the stable base stack during training-objective testing.

## Training-Objective Ablations Completed
### 1) Outcome credit placement
- Compared default outcome credit vs `outcome_credit_all_steps`.
- Result (fresh 10Q eval):
  - Control: `35.0%`
  - All-steps variant: `28.3%`
- Conclusion: `outcome_credit_all_steps` underperformed baseline in this paired test.

### 2) Reward-weight mix (valid remote-judge run)
- Baseline weights: `router=0.3, solver=0.5, outcome=0.2`
- Outcome-heavy weights: `router=0.1, solver=0.4, outcome=0.5`
- Result (fresh 10Q eval):
  - Baseline: `28.3%`
  - Outcome-heavy: `31.7%`
- Conclusion: outcome-heavy won and was promoted.

## Promoted Larger Follow-Up Run
- Run:
  - `router_solver_v2/experiments/outcome_heavy_50q_remotejudge_20260510_100507`
- Config:
  - `train_questions=50`, `G=6`, `epochs=1`, remote judge on
  - weights `0.1/0.4/0.5`
  - `use_answer_synthesis=on`
  - `plan_parse_repair=on`
  - `router_prompt_hardening=off`
- Training completion:
  - `50/50` questions
  - `300/300` valid rollouts
  - final train relaxed accuracy: `25.7% (77/300)`
  - checkpoints:
    - `checkpoint_epoch0_q50.pt`
    - `phase4_final.pt`

## Comparison vs Previous Large Run
- Previous large slim run (`slim_g6_20260508_153045`) ended around `21.8%` train accuracy.
- New promoted 50Q outcome-heavy run: `25.7%`.
- Directional gain: about `+3.9 pts` on training readout (not perfectly same scale, but positive).

## What Was and Was Not Exhausted
- Completed objective families:
  - outcome credit placement
  - reward mix baseline vs outcome-heavy
- Not yet completed:
  - no-router objective (`0.0/0.5/0.5`)
  - broader objective grid and larger paired objective sweeps

## Current State
- The 50Q training run is complete and checkpointed.
- Post-train eval/taxonomy stage was relaunched after creating missing `eval/` directory:
  - `router_solver_v2/experiments/outcome_heavy_50q_remotejudge_20260510_100507/eval/eval_trace.log`
- Final recommendation from this cycle:
  - keep outcome-heavy objective as active line
  - next objective test should be no-router vs outcome-heavy on a larger paired slice
