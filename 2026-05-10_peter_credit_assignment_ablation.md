# Credit Assignment Ablation: 10Q Quick Loop

## Setup

Paired comparison on top of the best current inference stack:

- `use_answer_synthesis = on`
- `plan_parse_repair = on`
- `router_prompt_hardening = off`

Only the training-side outcome credit assignment changed:

- control: default outcome credit on the last solver step only
- variant: `outcome_credit_all_steps = on`

Artifacts:

- control eval taxonomy: [control_default_credit/eval/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/credit_assignment_10q_20260510_005313/control_default_credit/eval/taxonomy_report.md:1)
- variant eval taxonomy: [variant_outcome_credit_all_steps/eval/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/credit_assignment_10q_20260510_005313/variant_outcome_credit_all_steps/eval/taxonomy_report.md:1)

## Result

### Training

- control final train relaxed accuracy: `15.0% (9/60)`
- variant final train relaxed accuracy: `26.7% (16/60)`

### Fresh traced eval after training

- control eval relaxed accuracy: `35.0%`
- variant eval relaxed accuracy: `28.3%`

Both evals had:

- `60/60` valid rollouts
- synthesis used for all final answers

## Interpretation

`outcome_credit_all_steps` improved the in-loop training metric, but produced a weaker final checkpoint on fresh stochastic evaluation.

That means:

- the variant fit the sampled training trajectories faster
- but that gain did not transfer to better post-training rollout quality on a fresh draw

## Taxonomy delta

### Control

- `wrong_numeric_final`: `15`
- `correct_number_in_trace_wrong_final`: `11`
- `copied_intermediate_as_final`: `8`
- `non_numeric_final_answer`: `4`
- `plan_endpoint_mismatch`: `1`

### Outcome credit on all steps

- `wrong_numeric_final`: `18`
- `correct_number_in_trace_wrong_final`: `11`
- `copied_intermediate_as_final`: `5`
- `non_numeric_final_answer`: `5`
- `plan_endpoint_mismatch`: `4`

## Bottom line

The all-steps credit variant reduced `copied_intermediate_as_final`, but it increased `wrong_numeric_final` and `plan_endpoint_mismatch`, and it lost on final traced eval accuracy.

Recommendation:

- keep default last-step outcome credit for now
- do not promote `outcome_credit_all_steps` based on this quick-loop result
