# Retrieval MOG Roadmap

**Date:** May 13, 2026

## Objective

Push the `hard`-branch retriever from “promising top-k signal” to a reliable structural retrieval subsystem.

## Current State

Probe-scale structural retriever:
- corpus: `2987` docs, `2130` queries, `1065` positive docs, `45` signature buckets
- stage-1 trained retriever:
  - same-target MRR: `0.2304`
  - same-signature MRR: `0.1896`
  - same-signature recall@1: `0.0165`
  - same-signature recall@5: `0.5165`
- stage-2 retriever after harder-negative mining:
  - same-target MRR: `0.2386`
  - same-signature MRR: `0.1962`
  - same-signature recall@1: `0.0151`
  - same-signature recall@3: `0.0649`
  - same-signature recall@5: `0.5414`

Interpretation:
- top-1 is still weak
- top-k is now real
- the next architecture should assume retrieval is a candidate generator, not a single-exemplar oracle

## Phase 1: Data Separation

Two-tier data policy:

1. `high-trust exemplar pool`
- source: clean `soft` solved traces
- use: graph memory contents
- goal: compact, reliable exemplars for prompt injection

2. `large retrieval-training pool`
- source: `partition.json + probe_rollouts.jsonl`
- use: retriever training only
- goal: scale, structural diversity, hard negatives

Rule:
- do not collapse these roles
- the probe bank is for retriever geometry
- the clean soft pool is for graph memory payload quality

## Phase 2: Retriever Training Loop

### `R2`
- done
- stage-2 hard-negative mining and retraining completed

### `R3`
Goal:
- mine negatives with the stage-2 retriever
- retrain again on the harder negatives

Expected signal:
- modest MRR gain
- stronger recall@3/5
- top-1 may still remain weak

### `R4`
Goal:
- add multi-positive training instead of one-positive-only contrastive pairs

Rationale:
- current structural labels are coarse
- many questions have more than one acceptable analog

## Phase 3: Structural Labels

Current weakness:
- operation signatures are still crude

Upgrade target:
- richer structural labels such as:
  - target quantity type
  - final transformation type
  - operation composition
  - remaining-vs-total-vs-ratio distinctions

Rationale:
- better labels should improve retrieval geometry more than more heuristic prompt massaging

## Phase 4: Top-k Memory Use

Current policy:
- keep `K > 1`

Implemented:
- light Hopfield-style rerank over retrieved top-k candidates

Next use of the Hopfield state:
- rerank retrieved candidates
- choose the top 1-2 exemplar payloads
- summarize top-k into a more compact hard prompt

Mathematical form:

\[
r_1, \dots, r_k = \mathrm{retrieve}(q)
\]

\[
z = \sum_{j=1}^{k} \mathrm{softmax}(\beta q^\top r_j)\, r_j
\]

Use `z` to:
- rerank candidates
- compress candidate bags
- condition hard-branch prompt construction

## Phase 5: MIL / Prototype Direction

This is not the next immediate step, but it is the next serious modeling direction.

Candidate design:
- treat the top-k retrieved candidates as a bag
- use multi-instance learning over the bag
- optionally learn a small prototype bank on top of retrieved candidate embeddings

Why:
- labels are noisy
- multiple analogs can be valid positives
- top-k is already much stronger than top-1

Sequencing:
1. stronger nonparametric retriever
2. stronger top-k Hopfield readout
3. MIL over candidate bags
4. only then prototype memory if needed

## Phase 6: Honest Validation

Every hard evaluation should separate:
- retrieval quality
- candidate-set quality
- solver use of retrieved exemplars

Track:
- recall@k
- whether a structurally useful analog appears in top-k
- whether the prompt-selected exemplar was actually good
- whether the solver still failed after seeing a good exemplar

## Active Task Queue

1. `R3`: stage-3 hard-negative mining with the stage-2 retriever
2. retrain stage-3 structural retriever
3. keep pulling clean `soft` exemplars from the long solved-pool run
4. rerun honest hard evaluation with:
- latest `soft` exemplars
- latest probe-trained retriever
- Hopfield top-k rerank
5. if retrieval still plateaus, implement multi-positive training

## Decision Rules

- if recall@5 keeps rising and honest hard accuracy rises, keep pushing graph memory
- if retrieval improves but hard accuracy stays flat, the solver or prompt contract is the bottleneck
- if both retrieval and hard accuracy flatten, improve labels before changing model class
