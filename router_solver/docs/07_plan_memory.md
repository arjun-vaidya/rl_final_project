# 07 · Plan Memory (Optional Extension)

Our **uniqueness angle**. Strictly additive — the core Router–Solver experiment runs without it; this doc describes how to bolt it on.

## What it is

A persistent key-value store of `(problem_embedding → successful_plan)` pairs. The Router retrieves the top-K most similar past plans at the start of each new problem and conditions on them when producing a new plan. Memory grows during training and (optionally) during inference.

This resurrects pillars 2 and 3 of the original project pitch (graph-based memory + unified train/inference optimization) and differentiates us from the closest prior work (Agent-as-Tool, ArCHer), neither of which has cross-problem plan memory.

## Why it's additive

- **No architectural change** to Router, Solver, or GRPO.
- **No new LoRA or gradient path** — memory is a read-side service, not a trained parameter.
- Router prompt gets one **optional section** that's omitted when memory is empty.
- Training loop adds exactly one post-rollout write hook.

If we run out of time, we cut doc 07 and the full experiment still stands.

## Design

### Components

| Piece | Choice | Notes |
|---|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (22M params) | Runs on CPU; ~10 ms per query |
| Store | Python dict + numpy array (upgrade to FAISS if slow) | A few MB at our scale |
| Key | 384-dim embedding of the question text | |
| Value | `{plan_json, final_answer, num_steps, reward}` | Enough to inspect later |
| Capacity | 10k entries (FIFO eviction thereafter) | We won't hit this in a 500-step run |

### Write policy

After each training rollout:

```python
if outcome_reward == 1.0 and is_valid_json(plan) and num_tool_errors == 0:
    memory.write(embed(question), {plan, answer, num_steps, reward})
```

Gate: only write **clean successes** — correct answer *and* structurally valid plan *and* no tool errors. This keeps stale/flaky plans out of memory.

### Read policy

At the start of each new problem:

```python
similar = memory.topk(embed(question), k=3, min_similarity=0.5)
# returns list of {plan, answer, ...} dicts; empty if memory is cold
```

Top-K=3 is a starting choice; will ablate.

### Router prompt (with memory)

```
You are a planner. Output a JSON plan of subgoals.

[IF similar is non-empty:]
Similar problems you have solved before:
1. Problem: {similar[0].question}
   Plan:    {similar[0].plan}
2. ...
[END IF]

Problem: {question}
Plan:
```

The `[IF ... END IF]` block is omitted entirely in Condition #4a (no memory) and when memory is cold at the start of training.

## Experimental conditions (additions to §05)

Extending the four conditions from [05](05_evaluation.md):

| # | Condition | Memory |
|---|---|---|
| 4a | Router–Solver, decomposed rewards | **no memory** (baseline for this extension) |
| 4b | Router–Solver, decomposed rewards | **random** retrieval from memory |
| 4c | Router–Solver, decomposed rewards | **similarity-based** retrieval (our method) |
| 4d (stretch) | Router–Solver, decomposed rewards | **RL-trained retrieval policy** |

Key comparisons:
- **4a vs 4c** — does memory + retrieval help at all?
- **4b vs 4c** — does *similarity* matter, or is it just "more in-context examples help"?

If 4b ≈ 4c, that's a clean negative finding: retrieval quality doesn't matter at our scale, exemplar quantity does. Still a publishable-for-a-course-project result.

## New metrics (on top of §05)

- **Retrieval-hit rate** — fraction of rollouts where at least one retrieved plan was structurally similar to the Router's eventual output.
- **Memory-growth curve** — size of memory over training steps (should plateau at problem-count × success-rate).
- **Accuracy vs memory size** — sweep memory caps at {100, 500, 2k, 10k} to see if the effect saturates.
- **Cold-start gap** — accuracy on the first 500 rollouts (memory empty) vs the last 500 (memory warm).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stale early plans poison memory forever | Write gate (clean successes only); optional quality-based eviction |
| Router memorizes rather than generalizes | Measure accuracy on held-out problems whose embedding has no close match (≥ 0.8) in memory |
| Memory just acts as extra context, similarity is irrelevant | Condition 4b is the explicit ablation for this |
| Embedding model biases retrieval toward surface lexical similarity | Sanity-check with a second embedder in the ablation if time permits |

## Code changes needed (additive)

```
router_solver/
├── src/
│   ├── memory/                    # NEW
│   │   ├── embedder.py
│   │   ├── store.py
│   │   └── retrieval.py
│   ├── agents/
│   │   └── router_solver_agent.py # one new kwarg: memory=None
│   └── training/
│       └── train_router_solver.py # one new hook: memory.write_if_success(...)
└── configs/
    ├── router_solver.yaml
    ├── router_solver_memory_random.yaml   # NEW
    └── router_solver_memory_sim.yaml      # NEW
```

No changes to `env/`, `rewards/`, or `eval/`.

## Timeline if we commit to this

Adds ~3 days to the base plan:

| Day | Work |
|---|---|
| +1 | Build `memory/` module + unit tests; wire into Router prompt |
| +2 | Run Condition 4c (similarity); sanity-check memory-growth curve |
| +3 | Run Condition 4b (random) as ablation; write up extension section |

Fits inside W2 if the core flat + Router–Solver runs (Conditions #2, #4a) finish by end of W1.

## Decision gate

**Only commit to Plan Memory if Condition #4a beats Condition #2 by end of W1.** If the core hierarchy isn't working, adding memory on top is just noise on top of noise — better to spend the time debugging the core.
