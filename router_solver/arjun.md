# Arjun's files

**Slice:** environment + flat baseline + evaluation harness + plan memory extension.

**Why this slice:** it's the end-to-end "infrastructure + measurement" track. You can ship Condition #2 (the flat baseline — our W1 gate) entirely from your own files. Peter can work on the hierarchical architecture in parallel without blocking you.

## Files you own

### Environment (W1, first)
- `src/env/gsm8k_loader.py` — load train / val / test, ground-truth extraction.
- `src/env/python_tool.py` — sandboxed subprocess Python exec with timeout.

### Flat agent + outcome reward (W1)
- `src/agents/flat_agent.py` — flat CoT+tool agent (Condition #2).
- `src/rewards/outcome.py` — numeric-match reward used by everything.

### Flat training (W1 — gate)
- `src/training/train_flat.py` — GRPO training loop for the flat agent.
- `configs/flat.yaml` — hyperparameters for Condition #2.

### Evaluation (W1–W2)
- `src/eval/evaluate.py` — accuracy, tool-call validity, tokens/episode, gradient-conflict diagnostic.
- `scripts/run_eval.sh` — CLI wrapper.

### Plan memory extension (W2 — only if W1 gate passes)
- `src/memory/embedder.py`
- `src/memory/store.py`
- `src/memory/retrieval.py`
- `configs/router_solver_memory_random.yaml`
- `configs/router_solver_memory_sim.yaml`

### Tests (alongside each module)
- `tests/test_env.py`
- `tests/test_rewards.py` → owns `test_outcome_*`
- `tests/test_memory.py`

### Infra
- `scripts/setup_env.sh`
- `requirements.txt`
- `.gitignore`
- `README.md` (you're the GitHub-repo owner)

## Interfaces Peter needs from you

These are the points where his code touches yours. Keep the shapes stable — if you need to change them, tell Peter first.

- `load_gsm8k_train() -> list[GSM8KProblem]` (dict with `question`, `answer`).
- `run_python(code) -> ToolResult(output, is_error, wall_time_ms)` — Peter's Solver calls this.
- `outcome_reward(trajectory_text, ground_truth) -> float` — Peter's router/solver rewards use this.
- `PlanMemory` class with `.retrieve(question) -> list[MemoryEntry]` and `.write_if_success(...) -> bool` — Peter's `RouterSolverAgent` takes this as a kwarg.

## Milestones / gates

| Gate | Date | What you ship |
|---|---|---|
| **G1 — env works** | Apr 23 | `pytest tests/test_env.py` passes; `run_python("4*2")` returns `"8"` |
| **G2 — flat baseline trains** | Apr 25 | One full `train_flat` run starts; loss decreases; val accuracy rises |
| **G3 — flat baseline hits target** | Apr 27 | Flat GRPO reaches ~75% on GSM8K test — this is the W1 gate for the whole project |
| **G4 — memory module ready** | May 2 | `tests/test_memory.py` passes; `PlanMemory` usable from Peter's agent |
| **G5 — memory ablation run** | May 6 | Condition #4c completed on val; comparison with #4a in hand |

## Spec references

- Environment: [docs/03_dataset.md](docs/03_dataset.md)
- Flat agent + outcome reward: [docs/04_design.md](docs/04_design.md)
- Evaluation: [docs/05_evaluation.md](docs/05_evaluation.md)
- Memory: [docs/07_plan_memory.md](docs/07_plan_memory.md)
