# POV: Answer-Target Failures vs Core Reasoning Failures

## Thesis

The `router_solver_v2` accuracy ceiling is not primarily explained by core reasoning failure.

On the traced `20Q x G=6` sample, only `45/120 = 37.5%` of sampled rollouts are best classified as genuine core reasoning misses. The larger share, `62/120 = 51.7%`, falls into non-core buckets: wrong answer targeting, copying an intermediate as the final answer, having the correct number somewhere in the trace but emitting the wrong final answer, or failing plan parsing.

That means the current `10.8%` rollout accuracy is likely being held down more by architecture / reward / output-contract problems than by the base model's inability to reason through GSM8K.

## Scope

This POV is based on the cached rollout trace and taxonomy pass here:

- traces: [rollout_traces.jsonl](/home/pvd2112/rl_final_project/router_solver_v2/experiments/taxonomy_q20_20260509_222707/rollout_traces.jsonl:1)
- taxonomy report: [taxonomy_report.md](/home/pvd2112/rl_final_project/router_solver_v2/experiments/taxonomy_q20_20260509_222707/taxonomy_report.md:1)
- taxonomy logic: [taxonomy.py](/home/pvd2112/rl_final_project/router_solver_v2/src/training/taxonomy.py:1)

The sample size is:

- `20` questions
- `6` rollouts per question
- `120` total rollouts
- `104` valid rollouts
- `16` invalid rollouts, all `plan_parse_failed`

Observed sampled rollout accuracy on this slice:

- exact accuracy: `13/120 = 10.8%`
- relaxed numeric accuracy: `13/120 = 10.8%`

## Bucket Definitions

### Core reasoning

`wrong_numeric_final`

Definition:

- the rollout is valid
- the final answer is numeric
- the trace does not clearly show the correct number already present
- the failure is not better explained by an endpoint mismatch or copying an earlier intermediate

Interpretation:

- the model likely did the math wrong, targeted the wrong quantity without obvious trace evidence, or made an unrecoverable reasoning error

### Non-core: answer-target / output / structure

`copied_intermediate_as_final`

Definition:

- the emitted final answer matches an earlier intermediate quantity rather than the true task target

Interpretation:

- the model is not necessarily failing to reason
- it is often failing to map the reasoning trace to the final answer contract

`correct_number_in_trace_wrong_final`

Definition:

- the correct numeric answer appears somewhere in a step answer or reasoning trace
- but the emitted final answer is still wrong

Interpretation:

- the model may already know the answer inside the trajectory
- the system fails to extract, synthesize, or privilege it correctly

`plan_endpoint_mismatch`

Definition:

- the last subgoal is not clearly answer-like
- the system still treats the last step answer as the global final answer

Interpretation:

- the decomposition endpoint is poorly aligned with the task target
- this is a system design failure more than a raw reasoning failure

### Non-core: parse / formatting

`plan_parse_failed`

Definition:

- router output could not be parsed into a valid plan

Interpretation:

- the model never got a usable execution scaffold
- this is not evidence that the solver could not solve the question

## Quantification

Primary bucket counts from the taxonomy:

- `wrong_numeric_final`: `45`
- `copied_intermediate_as_final`: `19`
- `plan_parse_failed`: `16`
- `correct_number_in_trace_wrong_final`: `15`
- `plan_endpoint_mismatch`: `12`

As shares of all `120` rollouts:

- `wrong_numeric_final`: `37.5%`
- `copied_intermediate_as_final`: `15.8%`
- `plan_parse_failed`: `13.3%`
- `correct_number_in_trace_wrong_final`: `12.5%`
- `plan_endpoint_mismatch`: `10.0%`
- already correct: `10.8%`

Collapsed into the core vs non-core split:

- correct now: `13/120 = 10.8%`
- core reasoning failures: `45/120 = 37.5%`
- non-core failures: `62/120 = 51.7%`

Where non-core is:

- `19 + 16 + 15 + 12 = 62`

## Projection Math

### Projection A: Answer-target / output fixes only

Assumption:

- treat the following buckets as recoverable without fundamentally improving reasoning:
  - `copied_intermediate_as_final`
  - `correct_number_in_trace_wrong_final`
  - `plan_endpoint_mismatch`

Recovered rollouts:

- `19 + 15 + 12 = 46`

Projected rollout accuracy:

- current: `13/120 = 10.8%`
- projected: `(13 + 46) / 120 = 59/120 = 49.2%`

Interpretation:

- if we fix only final-answer targeting and endpoint alignment, the sample suggests a plausible ceiling near `~49%` rollout accuracy

### Projection B: All non-core fixes

Assumption:

- all of Projection A
- plus eliminate `plan_parse_failed`

Recovered rollouts:

- `46 + 16 = 62`

Projected rollout accuracy:

- current: `13/120 = 10.8%`
- projected: `(13 + 62) / 120 = 75/120 = 62.5%`

Interpretation:

- if we also make plan generation robust enough to avoid parse dropouts, the sample implies a ceiling near `~62.5%`

## Question-Level View

Rollout-level accuracy is the main metric above, but the question-level view is even more revealing.

On these `20` questions:

- current questions with at least one correct rollout: `5/20 = 25%`
- if Projection A buckets are recovered: `18/20 = 90%`
- if all non-core buckets are recovered: `19/20 = 95%`

This strongly suggests that the model frequently produces at least one salvageable trajectory inside the `G=6` group, but the current system fails to cash that out into a correct final answer.

## Why These Buckets Should Be Treated as Non-Core

### 1. The system uses `last step answer = final answer`

Current implementation is structurally biased toward endpoint mistakes:

- agent rollout construction: [agent.py](/home/pvd2112/rl_final_project/router_solver_v2/src/agents/agent.py:1)

This works only if:

- the final subgoal really is the global answer target
- the final step answer is reliably the right scalar to emit

The taxonomy shows that assumption fails often.

### 2. Outcome credit is attached only to the last step

Current GRPO objective:

- training loss: [train.py](/home/pvd2112/rl_final_project/router_solver_v2/src/training/train.py:1)

The outcome term is applied only to `step_logps[-1]`.

That means:

- earlier reasoning can contain the correct number
- the model can still lose credit by emitting the wrong last-step quantity
- the system then treats this as a failed rollout even when the trace contains enough signal to recover

### 3. The taxonomy examples are visibly architectural, not purely cognitive

Representative categories:

- `correct_number_in_trace_wrong_final`
- `copied_intermediate_as_final`
- `plan_endpoint_mismatch`

These are exactly the kinds of failures expected when:

- decomposition endpoints are weak
- answer extraction is fragile
- final-answer synthesis is missing

They are not the same thing as "the model could not do the math."

## What This Does Not Prove

These projections are upper bounds on this sample, not guaranteed training outcomes.

Important caveats:

- the `20Q` sample is still small
- bucket labels are heuristic, not human-adjudicated
- some `wrong_numeric_final` cases may still contain hidden recoverable structure
- some `plan_endpoint_mismatch` or `correct_number_in_trace_wrong_final` cases may still mask deeper reasoning defects

So the right interpretation is:

- `~49%` is a plausible answer-target-only ceiling on this slice
- `~62.5%` is a plausible ceiling if all clearly non-core failures are removed
- actual realized performance after fixes could be lower if those fixes interact poorly with training

## Design Implications

### Highest-value interventions

1. Add a dedicated final-answer synthesis step.

Reason:

- many rollouts contain useful reasoning or the correct value, but the system emits the wrong final scalar

2. Stop assuming the last subgoal answer is the global answer.

Reason:

- `plan_endpoint_mismatch` is common enough to matter

3. Push correctness credit beyond the last step.

Reason:

- outcome reward currently supervises only the final step token path
- that is too weak when earlier steps already hold the right information

4. Harden router plan generation / parsing.

Reason:

- `16/120` rollouts died before execution because the plan never parsed

### Lower-priority relative to the above

- tuning the exact metric again
- model-size changes
- more overnight runs with the same structure and reward stack

Those may help later, but this sample says the bigger immediate gains are architectural.

## Recommended Next Patch Order

1. Add a final answer synthesis head / step over the full trace.

Minimal version:

- after step generation, prompt the solver once more with the full trajectory and ask for:
  - `Final answer: <number>`

2. Score outcome on synthesized final answer, not just the last subgoal answer.

3. Keep the existing answer parser, but use it on the synthesis output first.

4. Make plan JSON generation more constrained or more repair-tolerant.

5. Re-run the same `20Q` traced taxonomy for direct before/after comparison.

## Bottom Line

The current sample does not support the story that low accuracy is mainly caused by core reasoning failure.

The stronger story is:

- `37.5%` of rollouts are core reasoning failures
- `51.7%` are non-core failures caused by answer targeting, endpoint mismatch, or parse failure

That implies a credible near-term opportunity:

- `10.8% -> ~49.2%` if answer-target/output issues are fixed
- `10.8% -> ~62.5%` if all clearly non-core issues are fixed

That is the highest-signal path forward.
