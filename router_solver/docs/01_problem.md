# 01 · Problem

## Setting

An **agentic LLM** takes actions in a loop: calls tools, reads results, reasons, and returns an answer. The dominant training approach is **RLVR** — Reinforcement Learning with Verifiable Rewards: run the model, check the final answer against ground truth, give reward ∈ {0, 1}. This works well on single-step tasks. It breaks down on multi-step tool use.

## Gradient conflict: a worked example

> *A store sells apples at $2 and oranges at $3. Alice buys 4 apples and 5 oranges, pays with $50. Change?*

```
Step 1 (good)  → python("4 * 2")      → 8
Step 2 (good)  → python("5 * 3")      → 15
Step 3 (good)  → python("8 + 15")     → 23
Step 4 (wrong) → python("50 + 23")    → 73   ← meant to subtract
Final answer: $73                               ← WRONG
```

Under outcome-only RL:
- The whole trajectory gets reward **0**.
- The gradient pushes the policy away from **all four steps equally**, including the three correct ones.
- The reverse also happens: a lucky right answer through two canceling errors reinforces both errors.

This is **gradient conflict**: the learning signal on each intermediate action depends on the quality of later actions it has no control over. As trajectory length $T$ grows, success rate $\approx p^T$ drops exponentially, so almost every rollout has reward 0 and the signal becomes both sparse and misattributed.

## What we want

A reward structure that credits planning quality and execution quality **separately**, while still using the ground-truth outcome signal where it's reliable. The Router–Solver decomposition in [02](02_approach.md) is the proposed fix.
