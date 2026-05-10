## Non-Core Ablation Follow-Up

Author: Peter
Date: 2026-05-10

### Scope

This report covers the two next inference-side non-core ablations run on top of the current promoted SOTA stack:

- `use_answer_synthesis = on`
- `plan_parse_repair = on`
- `router_prompt_hardening = off`

Checkpoint used for all comparisons:

- `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_153045/phase4_final.pt`

Reference SOTA baseline on the same 10Q slice:

- `/home/pvd2112/rl_final_project/router_solver_v2/experiments/candidate_rerank_refined_10q_20260510_032637/sota_baseline/taxonomy_report.md`
- relaxed rollout accuracy: `31.7%`
- valid rollouts: `60/60`

### Ablations Run

#### 1. `answer_bearing_step_hint`

Artifacts:

- traces: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/answer_bearing_step_hint_10q_20260510_034332/rollout_traces.jsonl`
- taxonomy: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/answer_bearing_step_hint_10q_20260510_034332/taxonomy_report.md`

Result:

- relaxed rollout accuracy: `20.0%`
- valid rollouts: `60/60`

Primary failure buckets:

- `wrong_numeric_final: 20`
- `correct_number_in_trace_wrong_final: 13`
- `copied_intermediate_as_final: 8`
- `non_numeric_final_answer: 4`
- `plan_endpoint_mismatch: 3`

Interpretation:

- Hinting synthesis with the most answer-like subgoal did not improve final-target selection.
- It preserved validity, but accuracy fell materially versus the current SOTA baseline.
- The dominant remaining failure stayed core numeric failure, with no compensating reduction in answer-target mistakes.

#### 2. `trace_consistency_guard`

Artifacts:

- traces: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/trace_consistency_guard_10q_20260510_034332/rollout_traces.jsonl`
- taxonomy: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/trace_consistency_guard_10q_20260510_034332/taxonomy_report.md`

Result:

- relaxed rollout accuracy: `20.0%`
- valid rollouts: `60/60`

Primary failure buckets:

- `wrong_numeric_final: 17`
- `copied_intermediate_as_final: 16`
- `correct_number_in_trace_wrong_final: 12`
- `plan_endpoint_mismatch: 3`

Other notes:

- `final_answer_source_counts`: `{"synthesis": 57, "consistency_fallback": 3}`
- explicit non-numeric finals were removed, but only because the guard intervened on `3/60` rollouts

Interpretation:

- The guard in its current form is too weak to matter often.
- When it does fire, the fallback tends to over-select intermediate values.
- This shifted failure mass into `copied_intermediate_as_final` without improving top-line accuracy.

### Tandem Execution Constraint

I initially launched both ablations in parallel. That failed during synthesis with a hard CUDA OOM because two full model replicas plus batched synthesis do not fit on one L4.

Operational conclusion:

- true tandem execution is not viable for these trace-rollout ablations on the current single-GPU setup
- sequential queuing is the correct MATRIX-mode execution strategy

### Outcome

Neither ablation should be promoted.

Current SOTA remains:

- `synthesis + repair_fallback_only`

Specifically:

- keep `use_answer_synthesis = on`
- keep `plan_parse_repair = on`
- keep `router_prompt_hardening = off`
- do not promote `answer_bearing_step_hint`
- do not promote `trace_consistency_guard`

### Next Remaining Non-Core Quick Hits

The remaining feasible quick-hit non-core attempts are:

1. deterministic heuristic final selector
2. synthesis self-consistency over the fixed trace
3. guarded heuristic fallback

These are being queued sequentially in MATRIX mode after this report.
