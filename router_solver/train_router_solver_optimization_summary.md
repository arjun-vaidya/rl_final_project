# Router-Solver Training Script: Baseline Rollback + Optimization Diff Summary

This summary tracks changes in:
- Baseline: `HEAD:router_solver/src/training/train_router_solver.py`
- Current working tree: `/home/pvd2112/rl_final_project/router_solver/src/training/train_router_solver.py`

Generated on: 2026-05-04.

## 1) High-level outcome
The training logic was restored to the historical baseline and then only optimization/fidelity fixes were layered on top.

- Baseline structure (`rollout -> reward computation -> grouped advantages -> router+solver update`) is preserved.
- Added targeted execution-mode and batching optimizations to reduce training overhead and improve stability.

## 2) Diff summary by section

### A. Reward-mode handling
**File:** [train_router_solver.py](router_solver/src/training/train_router_solver.py)

- Baseline behavior: any non-`outcome_only` reward mode used router + decomposed solver rewards.
- New behavior: explicit branch for supported modes with fail-fast on invalid values.

```diff
-if reward_mode == "outcome_only":
-    r_router = outcome
-    r_steps = [outcome for _ in rollout.steps]
-else:
+if reward_mode == "outcome_only":
+    r_router = outcome
+    r_steps = [outcome for _ in rollout.steps]
+elif reward_mode == "decomposed":
     r_router = router_reward(rollout.router_output, final_ans, gt)
     r_steps = [solver_step_reward(s.tool_result, outcome) for s in rollout.steps]
+else:
+    raise ValueError(f"Unsupported reward_mode: {reward_mode}")
```

### B. Model mode and optional compile guard
**File:** [train_router_solver.py](router_solver/src/training/train_router_solver.py)

- Preserved baseline PEFT setup and dual adapters.
- Added:
  - explicit `model.train()` after construction
  - optional `torch.compile` behind environment toggle to avoid surprising runtime regressions

```diff
 model = get_peft_model(...)
 model.add_adapter(...)
 model.to(device)
+model.train()
+use_compile = os.getenv("ROUTER_SOLVER_TRAIN_COMPILE", "0")...
+if use_compile and device == "cuda":
+    try:
+        model = torch.compile(model)
+        print("[train] torch.compile enabled")
+    except Exception as e:
+        print(f"[train] torch.compile failed: {e}. Proceeding without compilation.")
```

### C. Rollout collection phase optimization
**File:** [train_router_solver.py](router_solver/src/training/train_router_solver.py)

- Baseline collected rollouts with standard forward tracking.
- New behavior runs all rollout sampling under `torch.inference_mode()` while keeping `model.eval()`.

```diff
-model.eval()  # deterministic forward (no dropout) during sampling
-records = []
-for qi, (q, gt) in enumerate(batch):
-    ...
+model.eval()  # deterministic forward (no dropout) during sampling
+records = []
+with torch.inference_mode():
+    for qi, (q, gt) in enumerate(batch):
+        ...
```

### D. Reference-policy scoring is precomputed once
**File:** [train_router_solver.py](router_solver/src/training/train_router_solver.py)

- Baseline computed each reference log-probability inside the router/solver loops, repeatedly re-entering disabled-adapter paths.
- New behavior:
  - enters `torch.no_grad()` once
  - computes all `reference_logprobs(...)` in one pass
  - caches by tuple keys `(id(ro), "router")` and `(id(ro), "solver", step_index)`

```diff
-optimizer.zero_grad()
-total_router_loss = torch.zeros((), device=device)
-total_solver_loss = torch.zeros((), device=device)
-
-for ... in records:
-    ... r_ref = reference_logprobs(...)
-    ...
-    ... s_ref = reference_logprobs(...)
-    ...
+ref_logprobs = {}
+model.eval()
+with torch.no_grad():
+    for _, ro, _, _, _ in records:
+        ref_logprobs[(id(ro), "router")] = reference_logprobs(...)
+        for si, s in enumerate(ro.steps):
+            ref_logprobs[(id(ro), "solver", si)] = reference_logprobs(...)
```

### E. Adapter-switching and loss accumulation
**File:** [train_router_solver.py](router_solver/src/training/train_router_solver.py)

- Baseline switched adapter repeatedly in the same loop and accumulated two running totals.
- New behavior:
  - sets router adapter once for all router terms
  - sets solver adapter once for all solver terms
  - accumulates per-trajectory terms in a list and calls one backward pass

```diff
-optimizer.zero_grad()
-total_router_loss = torch.zeros(...)
-total_solver_loss = torch.zeros(...)
-
-for ... records:
-    agent._set_adapter(router)
-    ... total_router_loss += ...
-    if ro.steps:
-        agent._set_adapter(solver)
-        ... total_solver_loss += ...
-loss = (total_router_loss + total_solver_loss) / len(records)
+optimizer.zero_grad(set_to_none=True)
+losses = []
+
+agent._set_adapter(router)
+model.train()
+for ... records:
+    ... losses.append(grpo_term(...) / len(records))
+
+agent._set_adapter(solver)
+for ... records:
+    ... losses.append(grpo_term(...) / len(records))
+
+loss = torch.stack(losses).sum() if losses else torch.zeros((), device=device)
 loss.backward()
```

## 3) Why these changes are valid/intent-preserving

- Core algorithmic path is unchanged: same reward contracts, same per-question grouping, same dual-objective GRPO update.
- Optimizations only reduce redundant work and improve numeric/stability control:
  - less recomputation for reference policy
  - clearer train/eval boundary
  - single backward for combined losses
  - safer config for `torch.compile`

## 4) Notes / caveats

- `ROUTER_SOLVER_TRAIN_COMPILE=1` remains opt-in.
- If any unexpected adapter context remains between phases, it is scoped to explicit `_set_adapter(...)` calls.
- If needed, next pass can add small helper comments around the new cache keys for future readability.
