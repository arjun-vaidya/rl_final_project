# Hierarchical RL + Graph Memory — Plan

**Date:** 2026-05-11
**Author:** Arjun Vaidya (with Peter Vail Driscoll)
**Course:** ORCSE6529 — Advanced RL final project
**Report due:** 2026-05-15

This document captures the plan for the hierarchical RL + graph memory direction — the original project pitch — built on top of the linear-reasoning GRPO baseline and probe-derived partition that already exist in this repo.

---

## 1. Motivation

The linear-reasoning GRPO baseline (`linear_reasoning/`, `grpo_linear_math_version/`) trains the same single-pass CoT policy on every question. The probe over Qwen2.5-Math-1.5B-Instruct on GSM8K-train (`data_probing/`) shows that this is wasteful:

- **88.4%** of questions are trivial — every rollout is correct, so heavy reasoning is pure overhead.
- **3.8%** are hard — every rollout is wrong, so light reasoning is hopeless.
- Only **7.8%** sit in the mixed band where added reasoning actually changes the outcome.

A single flat policy cannot make this distinction. A **hierarchical policy** can: a high-level router decides *how much reasoning to apply*, and a low-level solver carries out that reasoning. The probe partition gives oracle labels for the routing decision essentially for free.

---

## 2. System architecture

```
                        ┌─────────────────────────────────┐
                        │   Question (e.g. GSM8K item)    │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │             ROUTER               │
                     │  π_router(action | question)     │
                     │                                  │
                     │  Input:  question embedding      │
                     │          (+ optional features:   │
                     │           length, keyword flags) │
                     │  Output: P(light) , P(heavy)     │
                     └─────────┬──────────────┬─────────┘
                               │              │
                      light ◀──┘              └──▶ heavy
                               │              │
                               ▼              ▼
                ┌──────────────────┐  ┌──────────────────────┐
                │   LIGHT SOLVER   │  │     HEAVY SOLVER     │
                │  (single CoT)    │  │  ┌────────────────┐  │
                │                  │  │  │ retrieve K=5   │  │
                │  1 generation    │  │  │ from graph mem │  │
                │  ~500 tokens     │  │  └────────┬───────┘  │
                │                  │  │           ▼          │
                │                  │  │  ┌────────────────┐  │
                │                  │  │  │ self-consis.   │  │
                │                  │  │  │ vote of 4 CoT  │  │
                │                  │  │  └────────┬───────┘  │
                │                  │  │           ▼          │
                │                  │  │     final answer     │
                └────────┬─────────┘  └──────────┬───────────┘
                         │                       │
                         └──────────┬────────────┘
                                    ▼
                            ┌───────────────┐
                            │    Answer     │
                            └───────┬───────┘
                                    │
                                    ▼
                        reward = correct − λ·cost(action)
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                   update solver        update router
                   (token-level         (bandit-style
                    GRPO)                GRPO over actions)
```

**Two-level credit assignment:**
- The **solver** receives token-level credit via GRPO, exactly like the linear baseline.
- The **router** receives a single scalar reward per question (correct − cost), trained bandit-style with GRPO over its categorical action distribution.

---

## 3. Component details

### 3.1 Router

- **Architecture (MVP):** linear classifier head on top of frozen Qwen base embeddings. ~1.5M params.
- **Action space:** binary `{light, heavy}` initially. Extensible to more actions later.
- **Input features:** mean-pooled question embedding from the base model's last hidden state. Optional cheap hand-crafted features (length, digit count, "%" presence) concatenated.
- **Training in two phases:**
  1. **Supervised warm-start** on the probe partition (trivial → light, mixed/hard → heavy). ~30 min.
  2. **End-to-end RL** with reward = correct − λ·cost(action). ~6-8 h.

### 3.2 Light solver

- Existing `LinearReasoningAgent.rollout` — single CoT pass, ~500 tokens.
- Either the base model or the SFT-warm-started + GRPO checkpoint from the linear baseline.
- Cost: 1 unit.

### 3.3 Heavy solver

- **MVP:** self-consistency over 4 rollouts (temperature 0.8, majority-vote on the boxed answer).
- **+memory variant:** retrieve K=5 nearest sub-problems from graph memory, prepend them to the prompt as worked examples, *then* sample 4 rollouts.
- Cost: 4 units (the 4 rollouts dominate).

### 3.4 Graph memory

- **MVP form:** flat FAISS index over correct rollouts from `probe_rollouts.jsonl`.
  - Embed each `(question, reasoning_chain)` tuple with the base model's mean-pooled hidden state.
  - At query time, embed the new question, retrieve K=5 nearest tuples.
- **Population:** ~6,600 correct rollouts from the trivial bucket plus correct rollouts from mixed-bucket probes. Built once before training.
- **Future extension (out of scope for 4-day MVP):** structured graph with edge types (decomposition, derivation, analogy), but flat retrieval is enough for the report.

---

## 4. Training data flow

```
   Phase 0:  PROBE (already done)
   ──────────────────────────────────────────────────────────────
        GSM8K-train  ──▶  probe.py  ──▶  probe_summary.csv
                                          probe_rollouts.jsonl
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │  partition    │
                                          │  mixed:  583  │
                                          │  hard:   286  │
                                          │  trivial: 131 │
                                          └───────────────┘

   Phase 1:  GRAPH MEMORY POPULATION (~10 min)
   ──────────────────────────────────────────────────────────────
        probe_rollouts.jsonl
        (keep correct ones)
                │
                ▼
        embed (question)
        embed (final reasoning chain)
                │
                ▼
        ┌─────────────────────────┐
        │   FAISS / flat index    │
        │   ~6,600 correct        │
        │   (Q, reasoning)        │
        │   tuples                │
        └─────────────────────────┘

   Phase 2:  ROUTER WARM-START (~30 min)
   ──────────────────────────────────────────────────────────────
        partition labels   ──▶   X = question embeddings
        (oracle routing)         y = 0 if trivial, 1 otherwise
                                       │
                                       ▼
                                cross-entropy
                                       │
                                       ▼
                                ROUTER (linear head)

   Phase 3:  RL FINE-TUNE END-TO-END (~6-8 h)
   ──────────────────────────────────────────────────────────────
        for each question:
            sample G router actions  ──▶  rollouts per action
                                                │
                                                ▼
            reward = correct − λ·cost(action)
                                                │
                       ┌────────────────────────┴───────────┐
                       ▼                                    ▼
              GRPO update on router            GRPO update on solver
              (group over G actions)           (only on chosen path)
```

---

## 5. Example trajectories

```
   Q1: "What is 2+3?"
        │
        ▼
   ROUTER  →  P(light)=0.95  ●  P(heavy)=0.05
        │
        ▼
   LIGHT  →  "2+3 = \boxed{5}"   reward = 1.0 − 0·0 = +1.0
                                  ↑ correct, no cost penalty


   Q2: "A train leaves A at 60 mph and another from B at 80 mph,
        meeting in 3 hours. How far apart were A and B?"
        │
        ▼
   ROUTER  →  P(light)=0.20  P(heavy)=0.80  ●
        │
        ▼
   HEAVY  →  retrieve similar relative-speed problems from graph
          →  4 self-consistent rollouts, majority vote → 420
                                  reward = 1.0 − 0.1·1 = +0.9
                                  ↑ correct, paid for heavy path


   Q3 (hard): "Compound interest with quarterly compounding..."
        │
        ▼
   ROUTER  →  P(heavy) = 0.92  ●
        │
        ▼
   HEAVY  →  ...vote wrong         reward = 0.0 − 0.1·1 = −0.1
                                   ↑ router learns: even heavy
                                     can't solve some problems
                                     → may shift toward light on
                                     similar-feature hard ones
```

---

## 6. Timeline (4 days to report)

```
   Day 1 (May 12) — SCAFFOLD
   ┌─────────────────────────────────────────┐
   │ • Inspect router_solver_v2/ for reuse   │
   │ • Build graph memory (FAISS over probe) │  ~3 h
   │ • Build heavy solver (self-consistency) │  ~2 h
   │ • Smoke test: heavy vs light on 50 Qs   │  ~1 h
   └─────────────────────────────────────────┘
                       │
   Day 2 (May 13) — ROUTER WARM-START
   ┌─────────────────────────────────────────┐
   │ • Embed all partition questions         │  ~30 m
   │ • Train router (linear head) on oracle  │  ~30 m
   │ • Validate routing acc on held-out split│  ~1 h
   │ • Run inference-only routed system on   │
   │   GSM8K-test (no RL yet) → baseline #3  │  ~1 h
   └─────────────────────────────────────────┘
                       │
   Day 3 (May 14) — RL TRAIN
   ┌─────────────────────────────────────────┐
   │ • End-to-end GRPO on router (+ solver)  │  ~6-8 h
   │ • Tune λ (cost weight): 0.05, 0.1, 0.2  │
   │ • Save 3 variants                       │
   └─────────────────────────────────────────┘
                       │
   Day 4 (May 15) — EVAL + WRITEUP
   ┌─────────────────────────────────────────┐
   │ • Full GSM8K-test eval on all variants  │  ~2 h
   │ • Ablations: w/ vs. w/o graph memory    │
   │              w/ vs. w/o router warm-start│
   │              w/ vs. w/o RL fine-tune    │
   │ • Compile results table for report      │
   └─────────────────────────────────────────┘
```

---

## 7. Expected results table (for the report)

```
                          GSM8K-test       Avg compute      Δ vs base
                          accuracy         per question     (acc)
   ──────────────────────────────────────────────────────────────
   Base Qwen-Math-1.5B    XX.X%            1.0×             —
   GRPO (linear)          XX.X + 2-6       1.0×             +2-6
   SFT-warm + GRPO        XX.X + 5-10      1.0×             +5-10
   Always-heavy           XX.X + 8-15      4.0×             +8-15
   ROUTED (this work)     XX.X + 6-12      ~1.4×            +6-12
   ROUTED + graph mem     XX.X + 8-15      ~1.5×            +8-15
   ──────────────────────────────────────────────────────────────
                                            ▲                ▲
                                            │                │
                                heavy cost paid only         the headline
                                where it actually helps      number
```

**Thesis statement for the report:**
> We learn a router that decides when to deploy extra reasoning compute, achieving near-always-heavy accuracy at ~1.5× rather than 4× the cost — and an additional lift from a graph-memory retrieval module that conditions the heavy path on prior solved sub-problems.

---

## 8. Ablations to include

1. **Routing alone**: routed system *without* graph memory vs. always-light and always-heavy. Shows the router earned its keep on the compute/accuracy frontier.
2. **Graph memory effect**: heavy-with-memory vs. heavy-without (no routing). Shows memory helps independent of routing.
3. **Router training stages**: random router vs. supervised-only vs. supervised+RL. Shows the RL phase adds value.
4. **Cost weight λ**: sweep λ ∈ {0.05, 0.1, 0.2} to trace the accuracy/compute Pareto frontier.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Heavy solver doesn't beat light enough to justify routing | Medium | Inspect heavy-vs-light gap on the mixed bucket during Day 1 smoke test. If gap < ~5pp, switch heavy path to plan-then-solve or longer CoT (`max_cot_tokens=1024`). |
| Graph memory retrieval is noisy / unhelpful | Medium | Day 1 quick eyeball test: retrieve K=5 for 10 sample questions and check qualitatively. If garbage, fall back to no-memory heavy. |
| RL fine-tune destabilizes router warm-start | Low | KL anchor to warm-started router. Start with low LR (1e-5) and conservative λ. |
| Time crunch — can't finish all ablations by May 15 | High | Priority order: routed (no memory) → routed + memory → ablations. The first two are the headline; ablations can be partial. |

---

## 10. What's already in place

- `data_probing/` — probe results, partition derivation, analysis script. ✓
- `grpo_linear_math_version/` — GRPO baseline on the 1000-question partition. Currently running G=8 mb=4 on the VM, expected finish ~06:30 VM time. ✓
- `linear_reasoning/` — `LinearReasoningAgent` (reusable as the light solver), GRPO training loop (reusable for both router and solver updates). ✓
- `router_solver/`, `router_solver_v2/` — earlier scaffolding from the original project pitch. **TODO Day 1: audit what's reusable.**

---

## 11. Open decisions

1. **Should router and solver share a backbone?** Parameter-efficient (one LoRA, two heads) but couples gradients. Default: share backbone, separate LoRA adapters.
2. **What goes in the heavy path?** MVP is self-consistency over 4 rollouts. Alternatives: plan-then-solve, two-pass critique-revise, larger model fallback. Decide after Day 1 smoke test.
3. **How to evaluate compute cost?** Wall clock per question is the honest measure, but token-count is more portable. Report both.
4. **Should we train a separate solver per path?** No — too costly. Share the solver; the router just decides how many forward passes / what prompt scaffold to use.

---

## 12. Connection to the linear baseline

This plan does *not* replace the linear baseline — it builds on it. The linear-reasoning runs (the current `grpo_linear_math_version` run and any SFT-warm-start follow-ups) become **baselines #1 and #2** in the results table. The routed system is **method #1**, and routed + graph memory is the **headline method**. The story arc is: flat policy → hierarchical policy → hierarchical policy with external memory.
