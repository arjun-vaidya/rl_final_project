# Goal Description

The current GRPO training loop and `RouterSolverAgent.rollout` sampling are practically frozen due to a severe CPU bottleneck, which results in underwhelming GPU utilization. We need to implement aggressive optimizations to break this bottleneck.

Currently, `train_router_solver.py` generates `B * G` rollouts sequentially with a batch size of 1, and evaluates the GRPO backward terms in a sequential Python loop. This directly violates MANTRA optimization principles (specifically Class 8: Vectorized training to remove the Python overhead wall).

This plan proposes radical fixes including vectorized generation, batched forward/backward passes, FP8 utilization, and architectures inspired by the DeepSeek GRPO implementation (DeepSeekMath/DeepSeek-R1).

> [!IMPORTANT]
> **User Review Required**
> 1. **vLLM Dependency**: The standard DeepSeek GRPO architecture heavily relies on `vLLM` for the rollout phase (generation) and `PyTorch` for the training phase. Are you open to adding `vLLM` to the dependencies (`requirements.txt`), or should we implement batched generation natively in Hugging Face Transformers?
> 2. **FP8 Support**: Does your GPU hardware natively support FP8 (e.g., Hopper/Ada Lovelace)? If not, we will fallback to FP16/BF16 but still use the vectorized logic.

## Open Questions

- If we batch the Solver subgoals, we need to batch the environment execution (`run_python`). Are the tools in `run_python` thread-safe or safe to run concurrently via a persistent `multiprocessing.Pool` or `concurrent.futures`?
- Do you want to enforce strict numerical parity testing between the new batched rollout and the old sequential rollout, or is rough statistical parity acceptable given the temperature sampling?

## Proposed Changes

---

### `router_solver/src/agents/router_solver_agent.py`

#### [MODIFY] router_solver_agent.py
- **Vectorized Generation**: Introduce a `batched_rollout` method that accepts a list of `B * G` questions.
- **Left-Padding**: Use left-padding on the tokenizer to allow batched generation for the Router.
- **FP8 Inference**: If vLLM is not used, configure `model.generate` to use `torch.float8_e4m3fn` (where supported) or deep mixed precision for the generation phase.
- **Batched Solver Loop**: Instead of iterating over subgoals sequentially per rollout, group all active subgoals across the batch, perform a single batched generation step for the Solver, and execute the tools concurrently.

### `router_solver/src/training/train_router_solver.py`

#### [MODIFY] train_router_solver.py
- **Vectorize Rollout Call**: Replace the nested `for _ in range(G): ro = agent.rollout(...)` loop with a single `agent.batched_rollout(...)` call.
- **Vectorized Forward Passes**: Rewrite `teacher_forced_logprobs` and `reference_logprobs`. Currently, they compute log-probs for a single sequence. We will pad `prompt_ids` and `completion_ids` across the chunk and perform a single batched forward pass with proper `attention_mask`.
- **Compiler-Accelerated Fusion**: Apply MANTRA Optimization Class 10 by wrapping the GRPO KL-divergence and advantage loss computation (`grpo_term`) in `@torch.compile(mode="reduce-overhead")`.

### `router_solver/src/env/python_tool.py`

#### [MODIFY] python_tool.py
- **Persistent Python Workers (Radical Optimization)**: Currently, `run_python()` calls `subprocess.Popen(sys.executable)` for every tool call. With hundreds of solvers, this causes a massive CPU bottleneck due to process startup latency. We will replace this with a persistent, pre-warmed pool of Python worker processes (e.g., using `multiprocessing.Pool`). The workers will maintain an active Python interpreter and execute code via `exec()` communicating through IPC. This eliminates the startup tax and dramatically boosts CPU throughput.

## Verification Plan

### Automated Tests
- Run `pytest` on the updated batched log-prob computation to ensure numerical parity with the sequential `teacher_forced_logprobs`.
- Run a short training script `python -m src.training.train_router_solver --max_steps 5` and measure the wall-clock time per step, confirming that GPU utilization spikes and CPU overhead drops.

### Manual Verification
- Review the `wandb` logs for step-time reductions.
- Confirm via `nvidia-smi` or PyTorch Profiler that GPU compute kernels are saturated and there are fewer small kernel launches.
