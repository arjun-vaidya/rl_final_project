# Peter's files

**Slice:** Router–Solver architecture + decomposed rewards + hierarchical training (Conditions #3 and #4).

**Why this slice:** it's the "methods" track — the architectural contribution of the paper. You can prototype prompts, parsing, and the agent class without GPU while Arjun stands up the env. First GPU run happens once his `python_tool.run_python` is merged.

## Files you own

### Prompts + parsing (W1, first — no GPU needed)
- `src/utils/prompts.py` — Router, Solver, Flat prompt templates + builders (memory-on/off variants).
- `src/utils/parsing.py` — JSON plan parser, `<code>` extractor, answer extractor.

### Hierarchical agent (W1–W2)
- `src/agents/router_solver_agent.py` — Router + Solver with shared frozen base, two LoRAs, adapter swap, optional memory kwarg.

### Reward design (W1–W2)
- `src/rewards/router.py` — structure-gated Router reward.
- `src/rewards/solver.py` — decomposed per-step Solver reward.

### Hierarchical training (W2)
- `src/training/train_router_solver.py` — joint GRPO for both LoRAs. Conditions #3 (`reward.mode: outcome_only`), #4 (`decomposed`), #4a/b/c (via memory config).
- `configs/router_solver.yaml` — hyperparameters for #3 and #4.

### Tests (alongside each module)
- `tests/test_agents.py`
- `tests/test_rewards.py` → owns `test_router_*` and `test_solver_*`.

## Interfaces you need from Arjun

You **import** these — don't reimplement them:

- `src.env.gsm8k_loader.load_gsm8k_*`
- `src.env.python_tool.run_python(code) -> ToolResult` — your Solver calls this per step.
- `src.env.python_tool.looks_sensible(output) -> bool` — your Solver reward uses this.
- `src.rewards.outcome.outcome_reward(...)` and `extract_numeric_answer(...)`.
- `src.memory.retrieval.PlanMemory` (later; your agent accepts it as `memory: PlanMemory | None = None`).

## Interfaces Arjun imports from you

- `RouterSolverAgent` — agent class.
- `router_reward`, `solver_step_reward` — for eval's gradient-conflict diagnostic.
- `parse_plan_json` — eval uses it to analyze plan diversity.

## Milestones / gates

| Gate | Date | What you ship |
|---|---|---|
| **G1 — prompts + parsing tested** | Apr 23 | `pytest tests/test_rewards.py tests/test_agents.py -k parsing` passes on hand-authored examples |
| **G2 — agent rolls out end-to-end** | Apr 26 | `RouterSolverAgent.rollout(question)` returns a `HierTrajectory` with ≥1 subgoal executed (can use a dummy base at first) |
| **G3 — Condition #4 training starts** | Apr 29 | One `train_router_solver` run kicks off; joint loss decreases |
| **G4 — #4 vs #2 comparison** | May 3 | Val accuracy for Conditions #2, #3, #4 in hand — primary headline for the slides |
| **G5 — ablations A/B/C done** | May 10 | Reward-weight sweep, max-subgoals sweep, shared-vs-separate LoRA comparison |

## Decision gates that affect you

- **W1 gate (Apr 27):** if Arjun's flat baseline doesn't reach ~75%, we pause your training and debug together — no point training the hierarchical agent on top of a broken pipeline.
- **Memory gate (end of W2):** only if Condition #4 beats #2 do we commit to Plan Memory. If it doesn't, cut memory, lean into the ablations + gradient-conflict story.

## Spec references

- Approach: [docs/02_approach.md](docs/02_approach.md)
- Prompts, LoRA setup, reward functions, GRPO loop: [docs/04_design.md](docs/04_design.md)
- Conditions + ablations: [docs/05_evaluation.md](docs/05_evaluation.md)
- Memory integration (later): [docs/07_plan_memory.md](docs/07_plan_memory.md)
