# Hierarchical Pivot Plan (Revised)

## Goal
Build a routed reasoning system with three execution paths:

- `easy`: cheap single-pass CoT for easy questions
- `soft`: current best hierarchical heavy-v1 without retrieval
- `hard`: hierarchical heavy-v1 plus heterogeneous graph retrieval implemented as a Hopfield-style associative memory

This is a staged pivot. We do not start RL until the branches are real and separable.

## Current Evidence
- Global synthesis is a net negative. The heavy branch should not rely on `use_answer_synthesis`.
- The best current branch contract is:
  - answer-bearing final step,
  - local final answer per rollout,
  - question-level majority vote over rollout finals.
- Router prompt hardening was a regression.
- Candidate rerank did not show strong evidence of helping.
- Remaining dominant failure class is still `correct_number_in_trace_wrong_final`, which means the branch often computes the right value somewhere but does not preserve answer identity cleanly to the final output.

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

## Phase Execution Order
1. Freeze heavy-v1
2. Implement `easy`
3. Implement `hard` retrieval using heterogeneous Hopfield memory
4. Run `easy-only`, `soft-only`, `hard-only`, `oracle-routed`
5. Train supervised router
6. Only then consider RL

## Immediate Next Phase To Execute
Phase 1 and the minimal part of Phase 2:

1. Add an explicit `easy` branch using the linear CoT baseline contract.
2. Reuse the existing plan-memory idea from the older router stack, but change the representation to heterogeneous nodes with embedded memory patterns.
3. Keep the first hard-branch retrieval payload simple:
  - top retrieved solved question(s),
  - their plan(s),
  - optional final answer(s).
4. Inject retrieval into the router prompt for the `hard` branch only.

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
