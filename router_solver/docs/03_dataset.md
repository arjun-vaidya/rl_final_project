# 03 · Dataset and Environment

## Choice: GSM8K + Python tool

**GSM8K** — 8.5k grade-school math word problems (Cobbe et al., 2021).

- Train: 7,473 · Test: 1,319.
- Ground truth: integer after `####` — exact-match verifier.
- Each problem needs 2–8 arithmetic steps → the multi-step regime where gradient conflict matters.

## Why GSM8K

| Criterion | Why it fits |
|---|---|
| Multi-step | 2–8 sequential operations per problem |
| Verifiable reward | Deterministic numeric match, no LLM judge |
| Needs a tool | Small LLMs fail arithmetic like `1384 * 27` without Python |
| Cheap rollouts | ~1 s per rollout on a small model |
| Well-studied | Plenty of published numbers to sanity-check against |

## Why not the alternatives

| Alternative | Rejected because |
|---|---|
| MATH | Too hard for 1.5B — we'd study base-model ability, not hierarchy |
| MultiArith / SVAMP | Too easy — barely multi-step, no gradient conflict |
| BFCL / ToolBench | Reward needs an LLM judge — confounds our comparison |
| ALFWorld / WebArena | Setup cost alone eats the timeline |

## Python tool

Sandboxed Python execution. Input: a code string. Output: stdout or exception message, truncated to 256 chars. `subprocess` with a 5-second timeout, no filesystem/network. Adversarial code is not a concern here.

```
python("4 * 2")                           → "8"
python("import math; math.sqrt(16)")      → "4.0"
python("1/0")                             → "ZeroDivisionError: division by zero"
```

## Episode structure

**Router–Solver:** Router emits plan → for each subgoal, Solver emits code, env runs it, result appended to working context → parse final numeric answer → compare to ground truth.

**Flat baseline:** Model emits reasoning with interleaved `<code>…</code>` blocks; env executes each as it appears; model emits `<answer>X</answer>` → compare.

Both agents share the same **token budget per episode** (max 1024 generated tokens total) for a fair compute comparison.

## Splits

- **Train rollouts:** GSM8K train, 7,473 problems, sampled with replacement.
- **Val:** 200-problem held-out subset of train, for curves and hyperparameter decisions.
- **Test:** full GSM8K test (1,319). Evaluated once per run at the end. **No hyperparameter tuning on test.**

## Expected accuracy ballpark

| Setting | Expected |
|---|---|
| Qwen2.5-1.5B-Instruct CoT, no tool | ~55% |
| Qwen2.5-1.5B-Instruct + Python, prompted | ~70% |
| + flat GRPO, ~1 epoch | ~75% |
| + Router–Solver GRPO | **?? — this is the measurement** |

If the flat GRPO baseline doesn't hit ~75%, the training setup is broken — debug that *before* running the hierarchical comparison.
