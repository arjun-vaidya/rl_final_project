# Ablation Report: Answer-Target Fixes and Parser Split

## Scope

This report summarizes the completed ablation work on `router_solver_v2` after the rollout-trace taxonomy identified three major non-core failure modes:

- `copied_intermediate_as_final`
- `correct_number_in_trace_wrong_final`
- `plan_endpoint_mismatch`

and one important upstream execution failure:

- `plan_parse_failed`

The goal was to test whether accuracy is bottlenecked more by:

- answer-target / output architecture
- parser / router contract failures
- or core reasoning quality

## Key Artifacts

### 20Q answer-target suite

- baseline traces: [taxonomy_q20_20260509_222707/rollout_traces.jsonl](/home/pvd2112/rl_final_project/router_solver_v2/experiments/taxonomy_q20_20260509_222707/rollout_traces.jsonl:1)
- baseline taxonomy: [taxonomy_q20_20260509_222707/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/taxonomy_q20_20260509_222707/taxonomy_report.md:1)
- `plan_repair_only`: [answer_target_suite/plan_repair_only/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/answer_target_suite/plan_repair_only/taxonomy_report.md:1)
- `synthesis_only`: [answer_target_suite/synthesis_only/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/answer_target_suite/synthesis_only/taxonomy_report.md:1)

### 10Q parser split

- baseline: [parser_split_10q/baseline/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/baseline/taxonomy_report.md:1)
- `repair_fallback_only`: [parser_split_10q/repair_fallback_only/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/repair_fallback_only/taxonomy_report.md:1)
- `router_prompt_hardening_only`: [parser_split_10q/router_prompt_hardening_only/taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/router_prompt_hardening_only/taxonomy_report.md:1)
- `hardening_plus_repair`: intentionally aborted as non-viable before completion

## Code Changes Landed

### Architectural ablation toggles

- [agent.py](/home/pvd2112/rl_final_project/router_solver_v2/src/agents/agent.py:1)
  - `use_answer_synthesis`
  - `router_prompt_hardening`
  - `plan_parse_repair`
  - synthesis prompt / generation path
  - fallback plan repair path
- [config.py](/home/pvd2112/rl_final_project/router_solver_v2/src/utils/config.py:1)
  - new config flags for those toggles
- [main.py](/home/pvd2112/rl_final_project/router_solver_v2/main.py:1)
  - CLI wiring for all toggles

### GPU / inference optimizations

- repeated-prompt router fast path in [agent.py](/home/pvd2112/rl_final_project/router_solver_v2/src/agents/agent.py:1)
  - uses one `generate()` call with `num_return_sequences=G` for the router prompt
- `torch.inference_mode()` for generation paths in [agent.py](/home/pvd2112/rl_final_project/router_solver_v2/src/agents/agent.py:1)
- TF32 enabled in [main.py](/home/pvd2112/rl_final_project/router_solver_v2/main.py:1)
- model generation cache explicitly enabled in [agent.py](/home/pvd2112/rl_final_project/router_solver_v2/src/agents/agent.py:1)

### Trace / taxonomy support

- synthesis metadata added to rollout traces in [rollout_trace.py](/home/pvd2112/rl_final_project/router_solver_v2/src/utils/rollout_trace.py:1)
- taxonomy now records final-answer source counts in [taxonomy.py](/home/pvd2112/rl_final_project/router_solver_v2/src/training/taxonomy.py:1)

## 20Q Answer-Target Suite

### Result table

| Variant | Relaxed rollout acc | Valid rollouts | Notes |
|---|---:|---:|---|
| baseline | `10.8%` | `104/120` | original post-taxonomy benchmark |
| `plan_repair_only` | `9.2%` | `120/120` | removed parse failures, hurt correctness |
| `synthesis_only` | `20.0%` | `105/120` | best completed variant |

### Bucket shifts

#### Baseline

- `wrong_numeric_final`: `45`
- `copied_intermediate_as_final`: `19`
- `correct_number_in_trace_wrong_final`: `15`
- `plan_endpoint_mismatch`: `12`
- `plan_parse_failed`: `16`

#### `plan_repair_only`

- `wrong_numeric_final`: `30`
- `copied_intermediate_as_final`: `23`
- `correct_number_in_trace_wrong_final`: `37`
- `plan_endpoint_mismatch`: `19`
- `plan_parse_failed`: `0`

Interpretation:

- parser repair did exactly what it was supposed to do operationally
- it converted dead rollouts into executable ones
- but many of those recovered rollouts became answer-target failures rather than correct answers

#### `synthesis_only`

- `wrong_numeric_final`: `38`
- `copied_intermediate_as_final`: `15`
- `correct_number_in_trace_wrong_final`: `14`
- `plan_endpoint_mismatch`: `6`
- `plan_parse_failed`: `15`
- `non_numeric_final_answer`: `8`

Interpretation:

- synthesis clearly helps
- it cut `plan_endpoint_mismatch` in half
- it improved overall rollout accuracy from `10.8%` to `20.0%`
- but it introduced a new residual bucket: `non_numeric_final_answer`

### Conclusion from the 20Q suite

The strongest intervention tested so far is:

- `synthesis_only`

The weakest completed intervention is:

- `plan_repair_only`

So the main gain is coming from better final-answer targeting, not from parser hardening alone.

## 10Q Parser Split

This split was run specifically because the earlier `plan_repair_only` ablation conflated two changes:

- router prompt hardening
- parser repair fallback

The 10Q split isolated them.

### Result table

| Variant | Relaxed rollout acc | Valid rollouts | Interpretation |
|---|---:|---:|---|
| baseline | `20.0%` | `51/60` | fresh same-code baseline |
| `repair_fallback_only` | `18.3%` | `60/60` | slightly worse accuracy, much better validity |
| `router_prompt_hardening_only` | `3.3%` | `9/60` | catastrophic upstream regression |
| `hardening_plus_repair` | aborted | partial only | non-viable, not worth further GPU time |

### Bucket shifts

#### 10Q baseline

- `wrong_numeric_final`: `15`
- `copied_intermediate_as_final`: `13`
- `correct_number_in_trace_wrong_final`: `10`
- `plan_parse_failed`: `9`
- `plan_endpoint_mismatch`: `1`

#### `repair_fallback_only`

- `wrong_numeric_final`: `19`
- `copied_intermediate_as_final`: `13`
- `correct_number_in_trace_wrong_final`: `11`
- `plan_parse_failed`: `0`
- `plan_endpoint_mismatch`: `6`

Interpretation:

- repair fallback by itself is mostly fine
- it improves validity from `51/60` to `60/60`
- the accuracy cost is modest: `20.0% -> 18.3%`

#### `router_prompt_hardening_only`

- `plan_parse_failed`: `51`
- only `9/60` rollouts valid
- relaxed accuracy: `3.3%`

Interpretation:

- the stricter router prompt is the toxic change
- it is the primary source of the catastrophic validity collapse
- this confirms the earlier suspicion that the “parser fix” was not a clean parser-only change

### Conclusion from the parser split

The harmful component is:

- `router_prompt_hardening`

The acceptable component is:

- `repair_fallback_only`

So the right parser-side policy is:

- keep repair fallback
- drop router prompt hardening

## Main Findings

### 1. Answer synthesis is the highest-value completed intervention

Evidence:

- `10.8% -> 20.0%` on the 20Q benchmark
- strong reduction in endpoint mismatch
- clear improvement in final-answer targeting

### 2. Router prompt hardening is not viable

Evidence:

- `20.0% -> 3.3%`
- `51/60` invalid in the isolated 10Q split

### 3. Parser repair fallback is usable, but not a standalone accuracy fix

Evidence:

- it improves validity materially
- but it mostly migrates failures from `plan_parse_failed` into answer-target buckets

### 4. The data still supports the original thesis

The results continue to support:

- low accuracy is not mainly a metric artifact
- low accuracy is not explained purely by core reasoning failure
- architecture and answer-targeting are still the highest-leverage path

## Recommended Configuration Going Forward

Keep:

- `use_answer_synthesis = on`
- `plan_parse_repair = on` only if validity matters operationally
- router repeated-prompt fast path
- TF32 / inference-mode optimizations

Drop:

- `router_prompt_hardening = on`

### Preferred next ablation

The best next inference-side candidate is:

- `synthesis + repair_fallback_only`

but explicitly **without** router prompt hardening.

### Preferred next training-side ablation

Now that the parser split is resolved, the next training-side question should be:

- does `outcome_credit_all_steps` improve learning beyond `synthesis_only`?

That is the remaining major intervention from the original plan that has not yet been run to completion.

## Bottom Line

The suite resolved the ambiguity around the parser intervention.

The current evidence says:

- the good fix is `answer synthesis`
- the acceptable fix is `repair fallback`
- the bad fix is `router prompt hardening`

So the forward path is no longer ambiguous:

1. keep the synthesis line of work
2. keep repair fallback if we need the extra validity
3. remove prompt hardening from future runs
4. spend remaining experiment budget on training-side credit assignment, not more parser prompt tuning
