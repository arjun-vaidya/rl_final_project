# Hierarchical Router-Solver Architecture Status

**Date:** May 13, 2026

## Architecture

The current system has three branches:

1. `easy`
- single-pass CoT
- one final numeric answer
- cheapest branch

2. `soft`
- the main hierarchical router-solver branch
- decomposition + stepwise solving
- no global synthesis
- answer-bearing final-step repair
- question-level majority vote over local rollout finals

3. `hard`
- `soft` branch plus heterogeneous graph retrieval
- retrieved solved cases are injected into the hard prompt
- graph memory is currently nonparametric
- retriever is learned; memory contents are stored cases

## Current Read

- `easy` is implemented and useful as a cheap baseline.
- `soft` is the current production candidate.
- `hard` is still experimental.

The most important architectural conclusion is:

- the hierarchy is real
- the `soft` branch works
- the `hard` branch is no longer obviously fake, but it is not yet stable enough to make the main system depend on it

## Branch Status

### `easy`

Status:
- implemented
- used in binary and 3-way router smoke tests
- no serious standalone benchmark yet

Latest evidence:
- branch is functional
- useful as the cheap route for trivial/easy questions
- 10Q `B1` spot-check with `G=5`:
  - question-majority relaxed accuracy: `0.60`

Next steps:
- run `light-only` on the same 50Q slice used for branch separation
- measure question accuracy and compute cost against `soft`

### `soft`

Status:
- strongest working branch
- current mainline branch

Current best config:
- `use-answer-synthesis off`
- `plan-parse-repair on`
- `strict-answer-format off`
- `router-prompt-hardening off`
- `candidate-rerank off`
- answer-bearing final-step repair on
- deterministic majority vote over rollout local finals
- `solver_temperature=0.7`

Latest performance:
- 10Q diagnostic slice
  - question-level majority relaxed accuracy: `0.90`
  - question-level any-correct rate: `1.00`
  - rollout-level relaxed accuracy: `0.4833`
- this is the current `heavy-v1` winner

Latest ablation status:
- `H2` (`solver_temperature=0.4`, 10Q, `G=5`)
  - question-majority relaxed accuracy: `0.70`
  - decision: regression vs frozen baseline, drop
- `H3` (`solver_max_tokens=256`, 10Q, `G=5`)
  - question-majority relaxed accuracy: `0.40`
  - decision: clear regression, drop
- `B2` (`soft-only`, 10Q, `G=5`)
  - started under the old subprocess runner and then interrupted during refactor
  - still needs a clean completed single-process run

Current failure mode:
- still some `correct_number_in_trace_wrong_final`
- remaining errors now look like final-step drift or plan-endpoint corruption, not synthesis failure

Next steps:
- focus mainline effort here
- run outstanding `soft` ablations:
  - `solver_temperature=0.4`
  - `solver_max_tokens=256`
- run branch-separation eval:
  - `easy-only`
  - `soft-only`
  - oracle-routed
- consider mixed-bucket GRPO or solver-improvement work around this branch, not around `hard`

### `hard`

Status:
- implemented end to end
- graph memory and learned retriever are wired in
- still experimental

Graph / memory status:
- heterogeneous stored cases with node types like `Q/P/S/A/D`
- learned dense retriever over stored case embeddings
- nonparametric memory contents
- current “Hopfield” framing is best understood as associative retrieval over structured stored cases, not a fully parametric Hopfield network

Latest retrieval evidence:
- filtered structural retriever on the small filtered corpus:
  - same-target MRR: `0.2908`
  - same-signature MRR: `0.8333`
  - same-signature recall@1: `0.6667`
  - same-signature recall@3: `1.0`
- honest hard slice on `Q2/Q3/Q5`:
  - `2/3` correct
  - `Q2 Betty`: wrong
  - `Q3 Julie`: correct
  - `Q5 Mark`: correct
- confidence in `2/3` is low because `n=3`

Latest data-scale update:
- probe-derived retriever corpus from `partition.json + probe_rollouts.jsonl`:
  - `2987` docs
  - `2130` queries
  - `1065` positive docs
  - `1922` negative-only docs
  - `45` signature buckets
- this solves the retriever-scale problem much better than the old `experiments/` traces
- probe-trained structural retriever on that large corpus:
  - base `e5-small`
    - same-target MRR: `0.1284`
    - same-signature MRR: `0.0430`
    - same-signature recall@1: `0.0000`
    - same-signature recall@5: `0.0536`
  - trained structural retriever
    - same-target MRR: `0.2304`
    - same-signature MRR: `0.1896`
    - same-signature recall@1: `0.0165`
    - same-signature recall@5: `0.5165`
- interpretation:
  - retrieval encoder training now has a meaningful result at scale
  - this is a stronger signal than the earlier tiny filtered-corpus experiments

Current failure mode:
- retrieval geometry has improved
- remaining failures are now more about solver/prompt brittleness after retrieval than about the retriever being obviously useless

Next steps:
- use the probe-trained retriever checkpoint in the next honest `hard` evaluation
- keep graph memory exemplars cleaner and smaller than the probe bank
- do not promote `hard` to a first-class production route yet

## Recommended Focus

Mainline focus:
- `soft` router-solver improvements

Research track:
- `hard` graph retrieval improvements in parallel

Reason:
- `soft` is already strong enough to matter
- `hard` is promising but not yet trustworthy
- the best near-term system story is still `easy + soft`

## Router-Solver Objective Review

Current assessment:
- likely close to tapped out on inference-side heuristics
- not tapped out on training-objective improvements
- the highest-ROI remaining work is reward quality, credit assignment, and sampling

What the current trainer still gets wrong:
- router reward mixes downstream usefulness with an ad hoc plan-length prior
- step reward can collapse to weak proxies when judge supervision is off
- outcome credit is too coarse for a hierarchical system
- group-relative normalization still wastes many zero-variance groups

High-ROI, literature-backed changes still worth pursuing:
1. `DAPO`-style informative-group sampling and token-normalized policy updates
2. answer-bearing-step outcome credit instead of mostly last-step-only credit
3. stronger variance-aware filtering or normalization for router-solver groups

Not recommended:
- more strict-format or selector heuristics
- more router prompt hardening
- more decode-cap tuning as a mainline strategy

## DAPO Pivot Assessment

Relevant implementation:
- [README.md](/home/machina/pvd2112/rl_final_project/dapo_linear_math/README.md)

Truthful read:
- `dapo_linear_math` is a stronger RL recipe than the current router-solver GRPO loop
- it supersedes many generic GRPO-side cleanup ideas
- it does **not** supersede hierarchical-specific credit assignment work

Latest benchmark result:
- reduced but real `dapo_linear_math` benchmark on a 24-question `mixed_hard` mini partition with `G=5`
- train rollout accuracy: `0.5083`
- kept informative groups: `11`
- dropped all-correct groups: `6`
- dropped all-wrong groups: `7`
- self-consistency eval on `100` test questions:
  - majority-vote accuracy: `0.85`
  - pass@5: `0.94`
  - mean agreement: `0.906`

Interpretation:
- the DAPO outer loop is real and strong in this repo
- informative-group filtering is doing real work
- this benchmark is the correct reference point for router-solver integration

What DAPO already gives us:
- dynamic informative-group sampling
- mixed-hard bucket default
- token-level loss normalization
- stronger LoRA capacity
- stronger self-consistency evaluation

What DAPO does not solve for router-solver:
- hierarchical router reward design
- answer-bearing-step credit
- `correct_number_in_trace_wrong_final` as a decomposition/interface failure

Recommendation:
- stop treating the old router-solver GRPO loop as the long-term RL recipe
- treat `dapo_linear_math` as the new RL baseline
- if hierarchical training remains a core claim, port DAPO-style sampling and token-normalized training discipline into router-solver and pair it with hierarchical credit assignment

## Router-Solver DAPO Integration Decision

### Minimal decision matrix

Step 3 required clean rerun:
- patched router-solver
- `execution_branch=soft`
- `G=5`
- `informative_group_sampling=on`
- `outcome_credit_mode=all`
- same 10Q train slice
- same 10Q eval slice
- corrected eval path using true `G=5`

Corrected Step 3 result:
- train:
  - relaxed accuracy: `0.30`
  - kept informative groups: `7`
  - dropped all-correct groups: `0`
  - dropped all-wrong groups: `3`
- corrected eval:
  - rollout relaxed accuracy: `0.22`
  - question-majority exact accuracy: `0.30`
  - question-majority relaxed accuracy: `0.30`
  - question-any relaxed accuracy: `0.70`
- taxonomy:
  - `correct_number_in_trace_wrong_final`: `22`
  - `wrong_numeric_final`: `13`
  - `copied_intermediate_as_final`: `4`

Comparison against existing router-solver baselines:
- streamed baseline:
  - majority relaxed accuracy: `0.40`
- streamed `all_steps`:
  - majority relaxed accuracy: `0.50`
- DAPO-style informative-group sampling + `all_steps`:
  - majority relaxed accuracy: `0.30`

Decision:
- the DAPO-style informative-group port **failed** the main success criterion
- it did **not** beat the existing streamed `all_steps` baseline
- it also underperformed the simpler streamed baseline

Consequence for selective credit tuning:
- `answer_bearing_final` was implemented
- but the selective-credit comparison was intentionally not promoted after Step 3 failed
- once the stronger-trainer precondition fails, further credit tuning is not the high-ROI path

### Final architectural call

**Case B**

`DAPO-style filtering did not help enough inside hierarchy; keep DAPO as the linear reference but not the router-solver trainer.`

This is the current data-backed decision.

### Deeper router-solver trainer redesign attempt

Implemented redesign:
- keep streamed loss + checkpointing
- add component-wise rollout diagnostics:
  - `plan_endpoint_answer_like`
  - `answer_bearing_step_correct`
  - `final_relaxed_correct`
  - `correct_number_in_trace`
- add structured informative-group filtering:
  - keep groups with variance in final-answer correctness or answer-bearing-step correctness
- replace blanket step credit with dependency-local credit:
  - answer-bearing step
  - immediate predecessor step only

Result on the same 10Q / `G=5` train slice:
- train relaxed accuracy: `0.28`
- kept informative groups: `6`
- dropped all-correct groups: `0`
- dropped all-wrong groups: `4`

Comparison:
- minimal DAPO-style port:
  - train relaxed accuracy: `0.30`
  - kept informative groups: `7`
  - dropped all-wrong groups: `3`
- deeper redesign:
  - strictly worse on the available early signal

Decision:
- kill the redesign branch
- do not continue to eval or branch-separation verification on top of it
- the deeper redesign, in this scoped form, did not rescue the hierarchical trainer

What this means operationally:
- keep `dapo_linear_math` as the stronger RL reference implementation
- keep the router-solver code fixes:
  - streamed step loss
  - gradient checkpointing
  - train-time solver token cap
  - corrected `trace_rollouts`
- keep the additional component-diagnostic plumbing and the new credit/filter modes available for future research, but not as the selected training recipe
- do **not** freeze `informative_group_sampling` as the new steady-state `soft` trainer baseline
- do **not** spend more time on local hierarchical credit variants on top of this failed DAPO-port attempt
- if router-solver RL work resumes later, it should start from a deeper trainer redesign, not just this minimal informative-group filter port

## Routing Roadmap

Near-term routing goal:
- supervised router over `{easy, soft, hard}`

Order:
1. prove branch separation
2. train supervised router
3. only then consider RL routing

Important constraint:
- do not make `hard` a mandatory route until it wins on a larger honest slice

## Long-Term Research Roadmap

These are aspirational, not current blockers:

1. supervised 3-way router, then RL router
2. iterative multi-hop graph reasoning
3. end-to-end RL over graph retrieval/write policies
4. parametric Hopfield readout over retrieved case embeddings

Candidate parametric Hopfield readout:

\[
r_1, \dots, r_k = \mathrm{retrieve}(q)
\]
\[
z = \sum_{j=1}^{k} \mathrm{softmax}(\beta q^\top r_j)\, r_j
\]

Potential uses of `z`:
- rerank retrieved cases
- summarize retrieved cases into the hard prompt
- condition the router or hard branch

This is a plausible next-generation memory mechanism, but it should come after the current nonparametric graph path is better validated.

## Soft Router-Solver Checklist

### Goal
Stabilize and validate `soft` as the main hierarchical branch before doing more routing or graph promotion.

### Branch Definition
Frozen baseline:
- `use-answer-synthesis off`
- `plan-parse-repair on`
- `strict-answer-format off`
- `router-prompt-hardening off`
- `candidate-rerank off`
- answer-bearing final-step repair on
- deterministic majority vote over rollout local finals
- `solver_temperature=0.7`

Current headline:
- 10Q question-majority relaxed accuracy: `0.90`

### Work Block 1: Final Soft Ablations

#### `H2`
Goal:
- test whether `solver_temperature=0.4` reduces final-step drift further

Compare against:
- frozen `H1` baseline at `0.7`

Success criterion:
- beats `0.90` question-majority on the same 10Q slice
- or matches it with cleaner runtime or error behavior

Latest result:
- completed
- observed question-majority relaxed accuracy: `0.70` on 10Q with `G=5`

Decision:
- worse, drop

#### `H3`
Goal:
- test whether `solver_max_tokens=256` keeps quality while reducing runtime and late-generation corruption

Compare against:
- frozen `H1` baseline

Success criterion:
- no meaningful drop in question-majority accuracy
- lower runtime
- ideally fewer drift-like failures

Latest result:
- completed
- observed question-majority relaxed accuracy: `0.40` on 10Q with `G=5`

Decision:
- clear regression, keep current cap

### Work Block 2: Branch Separation

#### `B1`
Run:
- `easy-only`

Need:
- question accuracy
- compute per question

Purpose:
- establish cheap baseline

Latest result:
- completed
- observed question-majority relaxed accuracy: `0.60` on 10Q with `G=5`

#### `B2`
Run:
- `soft-only`

Need:
- same slice as `B1`
- question accuracy
- compute per question

Purpose:
- prove `soft` actually buys useful lift

Latest status:
- pending clean completion under the refactored single-process runner

#### `B3`
Run:
- oracle-routed upper bound

Rule:
- easy or trivial-like -> `easy`
- mixed or harder -> `soft`

Need:
- accuracy
- compute

Purpose:
- prove routing is worth building before doing more router work

Latest status:
- pending `B2`

Decision rule:
- if `soft` does not beat `easy`, stop and improve `soft`
- if oracle-routed is not materially better on compute or accuracy tradeoff, routing is not worth prioritizing yet

### Work Block 3: Soft Error Review
After `H2/H3/B1/B2/B3`, inspect failures for:
- final-step drift
- copied intermediate as final
- malformed plan endpoint
- correct-number-in-trace wrong-final

Purpose:
- decide whether the next gain should come from:
  - prompt or contract cleanup
  - token or temperature tuning
  - or GRPO or mixed-bucket training

### Work Block 4: Objective Upgrades

#### `O1`
Goal:
- measure how much router-solver training signal is being wasted on low-variance groups

Run:
- versioned diagnostics on the current `soft` checkpoint with stochastic group diagnostics and taxonomy

Purpose:
- decide whether objective work should start with DAPO-style sampling or with hierarchical credit assignment

#### `O2`
Goal:
- test `outcome_credit_all_steps` as a first coarse process-credit baseline

Run:
- versioned credit-assignment train/eval experiment

Purpose:
- establish whether broader credit alone helps before implementing answer-bearing-step credit

#### `O3`
Goal:
- decide whether to port DAPO-style sampling into router-solver training

Decision:
- if `O1` shows many dropped or zero-variance groups and `O2` is weak, prioritize the DAPO port
- if `O2` helps materially, prioritize hierarchical credit assignment first

Latest result:
- the minimal DAPO-style informative-group port was executed and cleanly evaluated at corrected `G=5`
- result: regression vs existing streamed router-solver baselines
- current decision: do not adopt this port as the stable `soft` trainer baseline

### Recommended Execution Order
1. `H2`
2. `H3`
3. choose final `soft-v1`
4. `B1`
5. `B2`
6. `B3`
7. failure review
8. `O1`
9. `O2`
10. decide DAPO-port vs answer-bearing-step-credit first

### Updated Checklist State
- `H2`: done, drop
- `H3`: done, drop
- `soft-v1`: still the existing `temp=0.7` baseline
- `B1`: done
- `B2`: pending
- `B3`: pending
- failure review: pending `B2/B3`
- `O1`: newly active
- `O2`: newly active
- DAPO benchmark: done
- router-solver DAPO port: done, regression
- final architecture decision: `Case B`
