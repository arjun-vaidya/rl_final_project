# Hierarchical Pivot Plan (Revised)

## Goal
Build a routed reasoning system with three execution paths:

- `easy`: cheap single-pass CoT for easy questions
- `soft`: current best hierarchical heavy-v1 without retrieval
- `hard`: hierarchical heavy-v1 plus heterogeneous graph retrieval implemented as a Hopfield-style associative memory

This is a staged pivot. We do not start RL until the branches are real and separable.

## Current Priority
Mainline focus is now the `soft` router-solver branch.

Reason:
- `soft` is the only clearly strong branch today
- `hard` retrieval is promising but still experimental
- the best near-term system story is `easy + soft`, with `hard` as augmentation research

## Current Evidence
- Global synthesis is a net negative. The heavy branch should not rely on `use_answer_synthesis`.
- The best current branch contract is:
  - answer-bearing final step,
  - local final answer per rollout,
  - question-level majority vote over rollout finals.
- Router prompt hardening was a regression.
- Candidate rerank did not show strong evidence of helping.
- Remaining dominant failure class is still `correct_number_in_trace_wrong_final`, which means the branch often computes the right value somewhere but does not preserve answer identity cleanly to the final output.
- `solver_temperature=0.7` is the current `soft` branch winner.
- The graph retriever is no longer obviously broken, but the `hard` branch is still not stable enough to be a first-class route.

## Branch Snapshot

### `easy`
- status: implemented, functional, lightly validated
- role: cheapest baseline path
- current action: keep for branch separation and supervised routing

### `soft`
- status: current production candidate
- best known 10Q metric:
  - question-majority relaxed accuracy `0.90`
  - question-any relaxed accuracy `1.00`
  - rollout relaxed accuracy `0.4833`
- current action: focus mainline effort here

### `hard`
- status: experimental retrieval-augmented path
- latest honest hard slice:
  - `2/3` correct on `Q2/Q3/Q5`
  - low confidence because `n=3`
- latest retrieval-scale update:
  - probe-derived retriever corpus now has `2987` docs, `2130` queries, `1065` positives, `1922` negatives, `45` signature buckets
- current action: keep as research track, not mainline dependency

## Non-Negotiable Constraints
- Keep apples-to-apples 10Q evaluation available at all times.
- Use question-level majority-vote accuracy as the branch headline metric.
- Do not reintroduce global synthesis as the default heavy-path combiner.
- Do not start RL before branch separation is demonstrated.

## Phase 0: Freeze Heavy-v1
Purpose: choose a stable heavy branch and stop tuning around the edges.

### Heavy-v1 contract
- `use-answer-synthesis off`
- `plan-parse-repair on`
- `router-prompt-hardening off`
- `candidate-rerank off`
- `strict-answer-format off`
- answer-bearing final-step repair on
- deterministic question-level majority vote over rollout local finals

### Acceptance criterion
- Best available 10Q branch result becomes the frozen heavy baseline.
- All later routing work uses this exact branch unless a change beats it clearly on the same 10Q protocol.

### Frozen Heavy-v1
- branch: `soft`
- solver temperature: `0.7`
- question-level majority-vote answer is the branch output contract

## Phase 1: Branch Definition
Purpose: make the routing problem explicit before touching RL.

### Branches
1. `easy`
- single-pass CoT
- one final boxed answer
- no decomposition
- no retrieval

2. `soft`
- heavy-v1
- decomposition + local-final voting
- no retrieval

3. `hard`
- heavy-v1
- retrieval-augmented router/solver prompts
- local-final voting

### Required output contract
Every branch must emit a single final numeric answer that can be compared directly at the question level.

## Phase 2: Heterogeneous Graph Memory
Purpose: implement retrieval only where it is architecturally justified.

### Scope
The graph memory is used only by the `hard` branch.

### Node types
- `Q`: question nodes
- `P`: plan nodes
- `S`: subgoal nodes
- `A`: final-answer nodes
- optional `T`: compact trace-summary nodes

### Edge types
- `Q -> P`
- `P -> S`
- `S -> S` (ordered step transitions)
- `S -> A`
- `Q -> A`
- optional similarity edges between `Q` nodes and between `S` nodes

### Embeddings
- each node stores an embedding
- question and subgoal text are embedded directly
- plan and answer nodes can use pooled text embeddings or aggregated child embeddings

### Hopfield-style retrieval
- query is the current question embedding
- memory returns the most associated pattern(s) from the heterogeneous graph
- returned artifacts must stay compact:
  - similar solved question
  - its plan
  - optional final answer
  - optional high-value subgoal(s)

### Important scoping rule
This is a retrieval module, not a new end-to-end reasoning engine.
We are not building multi-hop graph traversal or graph-controlled solving first.

## Phase 3: Oracle Routing
Purpose: prove that branch specialization is worth doing before learning a router.

### Oracle actions
- `easy/trivial -> easy`
- `mixed -> soft`
- `hard -> hard`

Oracle labels can come from:
- the probe partition, or
- branch winner on a fixed eval slice

### Evaluation
Run `easy-only`, `soft-only`, `hard-only`, and `oracle-routed` on the same slice.

### Decision rule
- If `soft` does not beat `easy` on the targeted subset, stop and fix branches before routing.
- If `hard` does not beat `soft` on hard questions, keep graph retrieval optional and do not force it into the main story.
- If `oracle-routed` is close to best-branch accuracy at lower compute, proceed to supervised routing.

## Phase 4: Supervised Router Warm-Start
Purpose: train routing without RL noise.

### Router output space
- `{easy, soft, hard}`

### Inputs
- question embedding
- optional cheap scalar features:
  - length
  - number count
  - keyword features for rates, totals, remaining quantities

### Training target
- oracle labels from Phase 3

### Success criterion
- routed inference beats `easy-only`
- routed inference approaches oracle-routed
- routed compute stays materially below `hard-only`

## Phase 5: RL Only If The Action Gap Exists
Purpose: optimize compute-aware routing after the branches are already real.

### Preconditions
- heavy-v1 is frozen
- branch separation is demonstrated
- oracle routing works
- supervised routing is directionally successful

### RL target
- router policy only first
- cost-sensitive objective:
  - `reward = correct - lambda * branch_cost`

### Not in scope initially
- joint RL over router + graph memory
- learned graph write/read policy
- learned multi-hop graph controller

## Router-Solver Experiment Matrix

| ID | Goal | Config | Metric | Status | Decision |
|---|---|---|---|---|---|
| `F0` | falsify synthesis layer | `use-answer-synthesis off`, majority over local finals | question majority accuracy | done | synthesis is harmful |
| `P0` | answer-bearing final step | endpoint repair + no synthesis | majority accuracy, taxonomy | done | keep |
| `P1` | router hardening | `router-prompt-hardening on` | majority accuracy | done | regression, drop |
| `P2` | candidate rerank | `candidate-rerank on` | rollout/majority accuracy | partial | not promising, drop for now |
| `P3` | review fixes | non-destructive endpoint repair + deterministic vote + in-memory summaries | 5Q spot-checks | done | keep |
| `H0` | heavy-v1 baseline | current patched base | majority accuracy on 10Q/50Q | done | superseded by `H1` |
| `H1` | lower solver drift | same as `H0`, `solver_temperature=0.7` | majority accuracy | done | winner; freeze as `heavy-v1` |
| `H2` | stronger drift control | same as `H0`, `solver_temperature=0.4` | majority accuracy | not run | backup |
| `H3` | shorter decode | same as `H0`, `solver_max_tokens=256` | majority accuracy + runtime | not run | useful |
| `B1` | light-only baseline | cheap CoT branch | question accuracy + compute | not run | needed for branch separation |
| `B2` | heavy-only baseline | `heavy-v1` | question accuracy + compute | partially known | needed on same 50Q slice |
| `B3` | oracle routed upper bound | `easy -> light`, `hard -> heavy` | accuracy + compute | not run | proves routing is worth it |
| `R1` | supervised router warm-start | simple classifier on oracle labels | routed accuracy + compute | smoke-tested only | real eval only after `B1-B3` |

## Outstanding Mainline Work

The mainline branch is now `soft`, not `hard`.

Outstanding `soft` work:
1. Run `H2` and `H3`.
2. Run `B1/B2/B3` on a consistent slice.
3. Decide whether mixed-bucket GRPO or solver-level prompt/contract changes give the best next gain.
4. Keep `hard` optional until it wins on a larger honest slice.

## Aspirational Routing + Memory Roadmap

Near-term:
1. supervised router over `{easy, soft, hard}`
2. RL router only after branch separation is demonstrated

Longer-term:
1. iterative multi-hop graph reasoning
2. end-to-end RL over graph retrieval or write policies
3. parametric Hopfield readout on top of retrieved case embeddings

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

This remains aspirational. It should not displace the current nonparametric graph pipeline until the simpler path is validated at larger scale.

## Phase Execution Order
1. Freeze heavy-v1
2. Implement `easy`
3. Implement `hard` retrieval using heterogeneous Hopfield memory
4. Run `easy-only`, `soft-only`, `hard-only`, `oracle-routed`
5. Train supervised router
6. Only then consider RL

## Immediate Next Phase To Execute
Mainline:

1. run the outstanding `soft` branch experiments (`H2`, `H3`)
2. run branch-separation experiments (`B1`, `B2`, `B3`)
3. freeze the `easy + soft` mainline story

Parallel research:

1. continue probe-derived retriever training
2. keep graph memory exemplars cleaner and smaller than the probe corpus
3. rerun `hard` on a larger honest slice once the retriever is retrained at scale

## What We Are Explicitly Not Doing
- not using global synthesis as the main final-answer combiner
- not using selector heuristics as the main rescue path
- not training the graph memory end-to-end before routing works
- not making retrieval mandatory for all questions
- not starting RL because “the architecture sounds right”

## Success Conditions For The Pivot
- `easy` is cheap and competitive on easy questions
- `soft` beats `easy` on the mixed slice
- `hard` plus graph retrieval beats `soft` on the hard slice
- oracle routing recovers most of the best-branch accuracy at lower average compute
- supervised routing is good enough that RL becomes an optimization step rather than a rescue step
