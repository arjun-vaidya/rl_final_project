# Router-Solver Optimization Report

#tags: optimization, memory, parity, training, live-data, router-solver

Date: 2026-05-05

## Scope
Applied memory-focused training-loop changes to `src/training/train_router_solver.py` and validated against historical run behavior.

## Implemented Changes

### 1) Teacher log-prob path hardened for memory + compatibility
- Updated `teacher_forced_logprobs` to attempt cache-disabled forward (`use_cache=False`) and fall back to default forward when unsupported.
- This reduces activation footprint on policy/reference scoring passes where KV cache is not useful.

### 2) Runtime optimization controls
- Added env flags in training script:
  - `ROUTER_SOLVER_GRADIENT_CHECKPOINTING` (default `1`)
  - `ROUTER_SOLVER_LOSS_CHUNK_SIZE` (default `0`/auto)
  - `ROUTER_SOLVER_MAX_STEPS`
  - `ROUTER_SOLVER_PARITY_VERIFY`
  - `ROUTER_SOLVER_TRAIN_COMPILE`
- Added runtime overrides for rollout shape:
  - `ROUTER_SOLVER_BATCH_SIZE`
  - `ROUTER_SOLVER_GROUP_SIZE`

### 3) Reference pass optimization and parity-safe objective equivalence
- Removed pre-allocation of a full in-memory `ref_logprobs` dictionary.
- Reference log-probs are now computed on-demand per record/chunk in the same adapter pass as policy scoring.
- This removes large-lived intermediate tensor retention while keeping the objective mathematically unchanged.

### 4) Loss chunking with live objective parity
- Reworked loss loop to compute router and solver GRPO terms in chunks.
- Each chunk does:
  - forward terms
  - local sum
  - `sum * scale` backward
- Added `_live_data_objective_no_grad` helper to compare:
  - unchunked full-batch objective
  - chunked objective
  on live rollout data (`ROUTER_SOLVER_PARITY_VERIFY=1`).
- Parity check validates objective reconstruction (`live_data_gap`) while keeping rollout data unchanged.

### 5) Reference-mode determinism
- Updated `reference_logprobs` to preserve and restore model train/eval mode around base-policy forward.
- Keeps reference scoring deterministic even when the training loop is in train mode.

## Validation State

### Historical OOM confirmation (before fixes)
- Old behavior crashed with CUDA OOM in `teacher_forced_logprobs` reference scoring path.
- Typical allocator state in that log showed only ~21 MiB free on a 21.96 GiB device, and allocation failure for a small 20 MiB tensor.

### Current runtime checks (current branch)
- Launch used (example):
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROUTER_SOLVER_MAX_STEPS=1 ROUTER_SOLVER_LOSS_CHUNK_SIZE=4 ROUTER_SOLVER_PARITY_VERIFY=1 ROUTER_SOLVER_TRAIN_COMPILE=0` 
  - with `configs/router_solver.yaml`
- Initial logs show successful startup and optimizer settings, then entered generation/training step but did not emit first `step=` before manual stop in this session.
- Current status therefore: OOM path appears fixed, objective parity helpers are in place, but full step-time benchmark + exact parity delta are not yet captured in log output.

## Risk / Impact
- Chunking is objective-equivalent but does not reduce rollout computation.
- Speed improvements are from reduced memory pressure and fewer long-lived intermediate tensors, not from reducing rollout count/length.
- Increasing batch size is now possible from a memory perspective, but step wall-time will scale with total rollouts and tokens; rollout/solver generation is still the dominant cost.

## Suggested Follow-up
1. Run one committed benchmark pass with:
   - `ROUTER_SOLVER_MAX_STEPS=1` (or a short warm-up horizon),
   - chosen `(B, G)` pair,
   - and parse final `[train][parity] live_data_gap` + step metrics.
2. If speed is still too slow, next high-ROI lever is rollout reduction via token/control budget (not model architecture change):
   - lower effective generated work per step or gate long generated trajectories.

## Latest adjustment (2026-05-05)

- Switched to **chunking-as-fallback**:
  - default training now uses one full backward pass (no chunked loss loops).
  - `ROUTER_SOLVER_LOSS_CHUNK_SIZE` now acts as an explicit opt-in, not a default behavior.
  - if CUDA OOM occurs during full-batch backward, it retries automatically with chunk size 4.
- Added explicit per-step progress prints:
  - `[train][step=<n>] start`
  - `[train][step=<n>] collected_rollouts ...`
  - `[train][step=<n>] step_time_sec=...`

Interpretation:
- If you want strict speed-first behavior, keep `ROUTER_SOLVER_LOSS_CHUNK_SIZE` unset/0.
- If you start hitting OOM again, set `ROUTER_SOLVER_LOSS_CHUNK_SIZE` to a small number (e.g., 4).
