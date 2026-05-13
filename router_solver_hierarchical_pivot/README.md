# router_solver_hierarchical_pivot

The ongoing pivot after V2 (`../router_solver_v2/`). Three branches (easy / soft / hard), the hard branch attaches a Hopfield-style memory module on top of a learned retriever. The current state and what is or is not working is in `../notes/project_plan_for_arjun_2026-05-12.md`; this README is the orientation note for the code.

## What this directory does

The router decides between three branches per question:

- **easy**: single-pass chain-of-thought. Same as `../linear_reasoning/`.
- **soft**: hierarchical Solver, no global synthesis, answer-bearing final-step repair, majority vote over local finals. This is the current production-candidate branch and reached ~90% question-level majority accuracy on a 10-question diagnostic slice with `solver_temperature=0.7`.
- **hard**: same hierarchical Solver but with a Hopfield-style memory module that retrieves structurally analogous solved problems from a corpus and conditions the plan on them. Currently still experimental.

The retrieval corpus is built from the solved-pool from earlier runs. A contrastive retriever is trained against structurally-positive pairs (problems with similar reasoning signatures) with hard-negative mining. Exact-question exclusion at retrieval time prevents leakage.

## Why this design

V2's failure taxonomy said 51.7% of failures are non-core (architectural). The three-branch design tries to route each question to the cheapest branch that can solve it, so the hard branch only carries problems that need its capacity. Memory retrieval is the differentiator: instead of asking a 1.5B model to discover reasoning patterns from scratch every time, retrieve a structurally similar solved problem and condition on it. This is closer in spirit to ReAct and KG-Agent than to vanilla GRPO.

What we found, captured in the plan note: the original retrieval objective was self-positive (retrieve alternate views of the same question), which improved same-question retrieval but did not teach the embedding space to retrieve structural analogies. Switching to a structural-positive objective improved retrieval metrics substantially but downstream hard-branch answer accuracy on the honest disjoint slice is still poor. The current bottleneck is corpus quality, not the retriever architecture.

## Subdirectory layout

```
linear_base/        copy of ../linear_reasoning/ used as the easy-branch backbone
src/                router + solver + memory implementation
  agents/agent.py     three-branch agent
  memory/             Hopfield graph + text embedder
  routing/            simple router classifier between the three branches
  rewards/            judge + shaper (inherited from V2)
  training/           train loop + diagnostics + taxonomy
  utils/              answer parsing, openai-compatible client
scripts/            corpus build, retriever training, hard-negative mining, diagnostics
ablation_suite/     ablation shell scripts (answer-target, credit, parser, robust matrices)
judge_ops/          judge VM deployment assets (same shape as V2)
experiments/        per-experiment rollout traces, taxonomy reports, retriever pipeline runs
```

## How to run

The main training entrypoint is `main.py` at this directory's root.

```bash
cd router_solver_hierarchical_pivot

# Train, three-branch agent, with retrieval enabled on the hard branch.
python main.py --mode train

# Diagnostic eval on the first 10 questions with G=6 (the fixed evaluation contract).
python main.py --mode eval --diagnostic-rollouts-per-q 6 --eval_questions 10

# Build the retrieval corpus from solved-pool traces.
python scripts/build_retrieval_corpus.py

# Mine hard negatives and train the contrastive retriever.
python scripts/mine_hard_negatives.py
python scripts/train_contrastive_retriever.py
```

Eval contract (must remain fixed for cross-run comparisons): diagnostics on the first 10 questions, `--diagnostic-rollouts-per-q 6`, extrapolate to 50Q by multiplying the 10Q result by 5.

## Important files

| Path | Purpose |
|---|---|
| `main.py` | CLI for the three-branch agent, train / eval / diagnostic modes |
| `src/agents/agent.py` | the three-branch agent (easy / soft / hard) |
| `src/memory/hopfield_graph.py` | Hopfield-style memory module, retrieval-time exact-question exclusion |
| `src/memory/text_embedder.py` | retriever embeddings (e5small + a learned projection) |
| `src/routing/simple_router.py` | branch selector |
| `src/training/diagnostics.py` | the diagnostic harness used for the fixed 10-question contract |
| `scripts/train_contrastive_retriever.py` | retriever training with structural-positive pairs |
| `scripts/build_retrieval_corpus.py` | corpus assembly from solved-pool traces |
| `pivot_plan.md` | the design doc that scoped this pivot |
| `linear_base/` | snapshot of `../linear_reasoning/` used as the easy-branch backbone |

## Notes and caveats

- This subtree is in flux. Numbers in W&B may be ahead of what is documented here. When in doubt, trust `notes/project_plan_for_arjun_2026-05-12.md` for what is or is not working.
- Hard-branch results that look strong on a memory-overlapping eval are typically leakage. Use the disjoint-memory probe in `experiments/disjoint_memory_*` to check.
- `linear_base/` is intentionally a snapshot, not a symlink. If you change `../linear_reasoning/src/`, you have to sync manually or the easy branch will drift.
- The retriever pipeline experiments under `experiments/retriever_pipeline_*` are sequential retries. The latest one (`retry3_20260512_181321`) is the current baseline.
- The judge VM and the V2 judge VM are the same shape; deployment assets are mirrored in `judge_ops/`. If you have V2's judge already up you can point this directory at it.
