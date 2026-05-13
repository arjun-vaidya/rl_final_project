# Project Plan: Hierarchical Router-Solver Pivot

**To:** ARJUN  
**From:** Peter Driscoll  
**Date:** May 12, 2026

## Executive Summary

We now have a working hierarchical router-solver implementation with three branches:

- `easy`: single-pass CoT
- `soft`: hierarchical solver with no synthesis, answer-bearing final-step repair, and majority vote over local finals
- `hard`: hierarchical solver augmented with heterogeneous Hopfield-style memory retrieval

The current system state is asymmetrical:

- The `easy` branch is implemented and stable.
- The `soft` branch is the strongest working branch and currently the production candidate.
- The `hard` branch is implemented end to end, but honest evaluation shows it is still experimental and not yet strong enough to route to in production.

The core finding from the last round of experiments is that the original hard-branch retrieval problem was not primarily an implementation gap. It was a retrieval quality problem. We fixed leakage, removed weak heuristics, trained a retrieval model explicitly, and added diagnostics. That improved structural retrieval metrics, but not enough to make the hard branch reliably solve disjoint evaluation cases. The remaining bottleneck is corpus quality and signal quality for retrieval training.

## Current Implementation Status

### 1. Easy Branch

Implemented and working.

- Single-pass chain-of-thought generation
- Simple final-answer contract
- Lowest-cost branch
- Serves as the baseline route for trivial and many mixed questions

### 2. Soft Branch

Implemented and working.

Current best configuration:

- no global answer synthesis
- answer-bearing final-step repair
- majority vote over local finals
- `solver_temperature=0.7`

Current read:

- this is the strongest reliable branch in the system
- on the main 10-question diagnostic slice, it reached approximately `0.90` question-level majority accuracy
- this is the branch to freeze as the current heavy-v1 baseline

### 3. Hard Branch

Implemented and working technically, but not yet validated as a useful route.

What exists:

- heterogeneous Hopfield-style memory module
- learned embedding support for memory keys
- exact-question exclusion to prevent leakage
- retrieved-case persistence in rollout traces
- honest disjoint-memory debug harness
- contrastive retriever training pipeline
- hard-negative mining pipeline
- retrieval diagnostics pipeline

What the latest experiments showed:

- earlier apparent hard-branch wins were partly caused by memory overlap
- after exact-question exclusion, the hard branch regressed sharply
- generic pretrained embeddings were insufficient
- explicit retriever training was necessary
- a structural-positive retrieval objective improved retrieval metrics more than self-positive matching
- downstream hard-branch answer accuracy is still poor on the honest disjoint slice

## What Has Been Falsified

The following approaches should not be treated as the main path forward:

- global synthesis as the heavy-branch combiner
- heuristic answer selectors as a primary fix
- generic off-the-shelf embeddings for graph retrieval
- retrieval heuristics as a substitute for learned retrieval
- self-positive retriever training as the main objective

## Retrieval Diagnosis

The key diagnostic conclusion is:

The original retrieval objective was ill-defined for the hard branch.

Self-positive training mostly learned to match alternate views of the same question. That helped same-question retrieval but did not teach the embedding space to retrieve structurally analogous math problems.

A structural-positive objective improved same-signature retrieval substantially, which means the embedding can be pushed in the right direction. However, honest downstream hard-branch performance remains weak, which indicates the current solved corpus is too small and too noisy to support a robust graph retriever.

In short:

- the embedding objective mattered
- the new objective helped
- the data quality is still not good enough

## Strategic Read

The project should proceed with the following framing:

- `easy` is stable and cheap
- `soft` is the current strongest branch and should carry the system in the near term
- `hard` remains a research branch focused on graph retrieval and hard-question specialization

For the graph specifically, the right training distribution is not broad uniform coverage. It should be biased toward `mixed + hard` questions, because those are the questions where memory and structural analogies matter. Trivial questions should mostly be used as negatives or calibration points, not as core graph exemplars.

## Next Two Hours: Concrete Execution Plan

The next execution block should focus on the graph corpus and retriever, not more router work.

### Phase 1: Filter the Graph Corpus

Build a filtered retrieval corpus from the solved pool and historical traces.

Filtering goals:

- keep only clean solved cases
- drop malformed plans
- drop duplicated or junk steps
- prefer answer-bearing final steps
- prioritize `mixed + hard-like` successful cases
- retain trivial questions primarily as negatives, not as graph exemplars

Deliverables:

- filtered corpus file
- signature distribution summary
- retained-case count by difficulty/type

### Phase 2: Rebuild and Diagnose the Retrieval Dataset

Rebuild the retrieval dataset from the filtered corpus and rerun diagnostics.

Diagnostics to track:

- same-question MRR
- same-target MRR
- same-signature MRR
- same-signature recall@1
- same-signature recall@3
- same-signature recall@5

Goal:

- verify that filtering improves structural signal instead of diluting it

### Phase 3: Retrain the Structural Retriever

Retrain the retriever using:

- structural positives only
- mined hard negatives
- no heuristic ranking

Goal:

- determine whether a cleaner corpus produces a better structural embedding space

### Phase 4: Rebuild Hopfield Memory on Learned Embeddings

After retraining, rebuild the hard-branch Hopfield memory from the learned embeddings.

Goal:

- ensure the graph retrieval layer reflects the new geometry directly

### Phase 5: Rerun the Honest Hard Slice

Rerun the targeted honest hard debug slice on the same disjoint questions.

Track:

- retrieved cases
- selected plan
- step answers
- final answer
- correctness against ground truth

Primary questions of interest remain the discriminative hard-debug slice where the current hard branch has been failing.

## Decision Gates

At the end of the next execution block:

### If retrieval metrics improve and honest hard accuracy reaches at least `1/3`

Continue investing in the graph branch.

### If retrieval metrics improve but honest hard accuracy stays `0/3`

The retriever is no longer the primary bottleneck. The hard solver contract or hard prompt integration becomes the next issue to debug.

### If retrieval metrics do not improve materially

The graph corpus is still too noisy, and the graph path should not yet be treated as a first-class route.

## Recommended Project Position

Near-term production posture:

- deploy `easy + soft` as the effective working hierarchy
- keep `hard` behind an experimental gate

Research posture:

- continue graph work, but treat it as a specialized hard-question memory program
- train the graph on `mixed + hard` structure, not broad easy-question coverage
- do not let graph complexity obscure the simpler fact that `soft` is currently the only strong heavy branch

## Closing

The hierarchical router-solver is now real at the system level. The `easy` and `soft` branches are working and interpretable. The `hard` branch is no longer blocked by missing infrastructure; it is now blocked by retrieval quality and corpus quality. That is useful progress, because the problem is finally localized.

The next step is not more router work. It is to clean the graph corpus, retrain the structural retriever, rebuild the Hopfield memory, and test the hard branch honestly again.
