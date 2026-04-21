# 02 · Approach

## Architecture

Split the agent into two policies acting at different levels of abstraction, implemented as two LoRA adapters on a single frozen base model.

```
                ┌─────────────┐
  question ───▶ │   ROUTER    │ ───▶ plan = [subgoal_1, subgoal_2, ...]
                └─────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  SOLVER  │  │  SOLVER  │  │  SOLVER  │
    │ subgoal_1│  │ subgoal_2│  │ subgoal_3│
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
          ▼             ▼             ▼
     tool call     tool call     tool call
                        │
                        ▼
                  final answer
```

- **Router** — sees the question, outputs a JSON plan: a list of subgoals, each with a short description and the intended tool.
- **Solver** — for each subgoal in turn, sees `(question, plan, previous_results, current_subgoal)` and outputs a single tool call plus a one-line rationale.

Adapter swapping between Router and Solver is cheap (memcpy-level via `peft`).

## Why this reduces gradient conflict

With decomposed rewards, the four outcome/plan/execution combinations become distinguishable:

| Outcome | Router reward | Solver reward | Flat outcome reward |
|---|---|---|---|
| Good plan, good execution, right answer | +1 | +1 | +1 |
| Good plan, bad execution, wrong answer | **+1** | 0 | 0 (everything punished) |
| Bad plan, good execution, wrong answer | 0 | **+1** | 0 (everything punished) |
| Bad plan, bad execution, lucky right answer | 0 | 0 | **+1** (everything rewarded!) |

The middle two rows are the gradient-conflict cases: the flat agent misattributes blame, while the hierarchical agent routes credit to the component that actually succeeded. The last row shows the flat agent *reinforcing bad behavior* on lucky outcomes — Router–Solver rewards are structural enough that lucky outcomes don't propagate as cleanly.

## Compared to flat CoT-with-tools

A flat agent would produce the same underlying token sequence (reasoning + `<code>…</code>` blocks interleaved). The difference is **where the reward signal attaches**: one signal for the flat agent, N+1 signals for the hierarchical one (one per subgoal + one on the final outcome). **This is the isolated variable we're testing** — the computation is the same; only the credit assignment changes.

## Extension point: Plan Memory

The architecture has one deliberate extension hook. The Router's prompt includes an **optional** section for retrieved plans from similar past problems. When that section is empty, behavior is identical to the core design above. When populated, the Router can condition on plans that previously succeeded — our uniqueness angle. See [07_plan_memory.md](07_plan_memory.md). Everything in the rest of this doc applies whether or not memory is enabled.
