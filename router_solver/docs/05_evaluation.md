# 05 · Evaluation Plan

## The question

> Does hierarchical decomposition (Router–Solver) improve sample efficiency and final accuracy over a flat GRPO baseline, when matched on base model, tool, and compute?

## Four conditions

All matched on base model (Qwen2.5-1.5B-Instruct), tool (Python), compute (~24 GPU-hours), and eval set (GSM8K test).

| # | Condition | Architecture | Reward | Purpose |
|---|---|---|---|---|
| 1 | **SFT baseline** | Flat, prompted | — (no RL) | Anchor: how far does prompting + tool get us? |
| 2 | **Flat GRPO** | Flat CoT + tool | outcome only | Standard RLVR baseline |
| 3 | **Router–Solver, outcome-only** | Hierarchical | outcome only, shared | Isolates architecture effect |
| 4 | **Router–Solver, decomposed** | Hierarchical | Router + Solver + outcome | Full method |

**Key comparisons:**
- **#2 vs #4** — does our full method beat the standard baseline?
- **#3 vs #4** — does reward decomposition add value beyond architecture?
- **#2 vs #3** — does hierarchy help even without decomposed rewards?

### Optional extension: Plan Memory (Conditions #4a/b/c)

If the core Router–Solver beats the flat baseline by end of W1, add a memory layer. See [07_plan_memory.md](07_plan_memory.md) for the full design. At a glance:

| # | Condition | Memory |
|---|---|---|
| 4a | Router–Solver, decomposed rewards | none (= #4 above) |
| 4b | Router–Solver, decomposed rewards | random retrieval |
| 4c | Router–Solver, decomposed rewards | similarity-based retrieval |

Comparison 4a vs 4c asks whether cross-problem memory helps; 4b vs 4c isolates whether *similarity* matters or just extra in-context exemplars.

## Metrics

**Primary:** GSM8K test accuracy (exact numeric match).

**Secondary:**
- Sample efficiency (accuracy vs training step — learning curves).
- Tool-call validity rate.
- Average steps / tokens per episode.

**Gradient-conflict diagnostic:**
- **Advantage-correlation:** for each trajectory, correlate per-step "quality" (tool succeeded + sensible output) with per-step advantage. Lower = more conflict.
- **Wasted-update fraction:** fraction of correct intermediate steps sitting inside reward-0 trajectories (i.e., penalized).

Prediction: conflict is highest in #2, lowest in #4. If this prediction fails, that's a story worth telling too.

## Ablations

| # | Variant | What we learn |
|---|---|---|
| A | Solver reward weights: (0.3 / 0.2 / 0.5) vs alternatives | Sensitivity to reward-weight choices |
| B | Max subgoals: 3 / 6 / 10 | Does longer planning help? |
| C | One shared LoRA vs two separate | Do we actually need two adapters? |
| D (stretch) | Router reward from structure+diversity only | Does outcome-coupling cause Router collapse? |
| E (stretch) | Curriculum: easy → hard | Does it mitigate gradient conflict on harder problems? |

Committing to A, B, C in the report.

## Qualitative analysis

- **Failure-mode gallery:** 5–10 side-by-side examples (flat fails / hierarchical succeeds, and vice versa).
- **Plan diversity:** number of distinct plan structures per problem across seeds.
- **Subgoal–execution alignment:** LLM-judge check on 100 trajectories — when handed subgoal X, does Solver code attempt X?

## Failure modes to monitor during training

- Router emits degenerate plans (always 1 step, always identical) → detect via plan-diversity metric.
- Solver ignores the subgoal, tries to solve the whole problem in one shot → detect via subgoal-alignment check.
- Router reward is gamed by any structurally-valid plan → detect by checking whether Router-reward improvement correlates with final accuracy.

## What counts as success

The rubric explicitly accepts negative results if the analysis is sharp. Three outcomes:

**Strong positive:** #4 beats #2 by ≥ 3 points on GSM8K test, gradient-conflict diagnostic moves as predicted, ablation A shows robustness.

**Weak / mixed:** #4 gain < 3 points, but gradient-conflict metric moves correctly and ablations expose *which* component matters (e.g., "architecture alone does the work, reward decomposition adds little"). Still a good report.

**Negative:** #4 ≤ #2. Post-mortem carries the report: Solver ignoring subgoals? Router degenerate? Qualitative analysis + hypothesis-refutation is the story.

## Timeline

| Week | Work | Gate |
|---|---|---|
| W1 (Apr 21–27) | Env, tool, flat baseline end-to-end (Condition #2) | #2 ≈ 75% on GSM8K |
| W2 (Apr 28 – May 3) | Router–Solver impl, run Conditions #3 and #4, slides | #4 has one completed run |
| W3 (May 4–15) | Presentation, ablations A/B/C, write report | Report submitted |

If a gate fails, **de-scope before adding scope** — drop ablations, shrink base model, cut to a GSM8K-Easy subset.

## What we will not claim

- No state-of-the-art claim.
- No generalization claim beyond what we test.
- If #2 vs #4 is within noise, we say so — no spin.
