"""
Joint GRPO training for the Router + Solver LoRA adapters.

Implements the loop described in docs/04_design.md §96-118:

    for step in range(num_steps):
        questions = sample_batch(train_set, B)
        # G rollouts per question → B*G trajectories
        ... compute router_reward, solver_step_reward ...
        router_loss = grpo_loss(router_rewards, grouping=by_question)
        solver_loss = grpo_loss(solver_rewards, grouping=by_question)
        (router_loss + solver_loss).backward()
        optimizer.step()

GRPO specifics:
  - Advantages are per-question group z-scores (no critic).
  - KL penalty β uses the base model (LoRA disabled) as the reference policy,
    via the approximate KL estimator exp(r) - 1 - r  where r = logπ - logπ_ref.
  - Router loss covers router-generated tokens under the router adapter.
    Solver loss covers the solver-generated tokens of every subgoal, under
    the solver adapter. Gradients flow independently to each adapter's LoRA
    since only the active adapter is engaged during each forward.

Supports both Condition #3 (reward_mode: "outcome_only") and #4 ("decomposed").
"""
import os
import argparse
from collections import defaultdict
from typing import List, Tuple
from time import time
from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# Load environment variables
load_dotenv()

from src.rewards.router import router_reward
from src.rewards.solver import solver_step_reward
from src.rewards.outcome import outcome_reward
from src.agents.router_solver_agent import Rollout


# --------------------------------------------------------------------------- #
# Reward / advantage helpers
# --------------------------------------------------------------------------- #

def compute_rewards(rollout: Rollout, gt: int, reward_mode: str):
    """
    Returns (router_reward_scalar, solver_reward_per_step, outcome).
    For outcome_only: everyone gets the same {0,1} outcome.
    For decomposed: router_reward gates on plan validity; solver gets 0.3/0.2/0.5.
    """
    final_ans = rollout.final_answer or rollout.router_output
    outcome = outcome_reward(final_ans, gt)

    if reward_mode == "outcome_only":
        r_router = outcome
        r_steps = [outcome for _ in rollout.steps]
    elif reward_mode == "decomposed":
        r_router = router_reward(rollout.router_output, final_ans, gt)
        r_steps = [solver_step_reward(s.tool_result, outcome) for s in rollout.steps]
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    return r_router, r_steps, outcome


def group_normalize(values: List[float]) -> List[float]:
    """Standard GRPO group-relative normalization. Zero advantage if all equal."""
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return [0.0] * len(values)
    return ((arr - mean) / (std + 1e-8)).tolist()


# --------------------------------------------------------------------------- #
# Log-prob + KL helpers
# --------------------------------------------------------------------------- #

def batched_teacher_forced_logprobs(
    model,
    prompt_ids_list: List[torch.Tensor],
    completion_ids_list: List[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    """Return per-token log-probs in a batched pass. Grad flows."""
    if not prompt_ids_list:
        return []
    
    inputs_list = [torch.cat([p.to(device), c.to(device)], dim=0) for p, c in zip(prompt_ids_list, completion_ids_list)]
    max_len = max(t.size(0) for t in inputs_list)
    
    pad_id = getattr(model.config, "pad_token_id", 0)
    if pad_id is None: pad_id = 0
    
    input_ids = torch.full((len(inputs_list), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(inputs_list), max_len), dtype=torch.long, device=device)
    
    for i, t in enumerate(inputs_list):
        input_ids[i, :t.size(0)] = t
        attention_mask[i, :t.size(0)] = 1

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, L, V]

    log_probs_list = []
    for i, (p, c) in enumerate(zip(prompt_ids_list, completion_ids_list)):
        p_len = p.size(0)
        c_len = c.size(0)
        if c_len == 0:
            log_probs_list.append(torch.zeros(0, device=device))
            continue
            
        shift_logits = logits[i, p_len - 1 : p_len - 1 + c_len, :]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        lps = log_probs.gather(-1, c.to(device).unsqueeze(-1)).squeeze(-1)
        log_probs_list.append(lps)
        
    return log_probs_list


def batched_reference_logprobs(
    model,
    prompt_ids_list: List[torch.Tensor],
    completion_ids_list: List[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    """Same as above but with all LoRA adapters disabled."""
    was_training = model.training
    if was_training:
        model.eval()
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                out = batched_teacher_forced_logprobs(model, prompt_ids_list, completion_ids_list, device)
        else:
            out = batched_teacher_forced_logprobs(model, prompt_ids_list, completion_ids_list, device)
    out = [o.detach() for o in out]
    if was_training:
        model.train()
    return out

def grpo_term(
    policy_lp: torch.Tensor,
    ref_lp: torch.Tensor,
    advantage: float,
    beta: float,
) -> torch.Tensor:
    """
    Per-rollout loss: -(advantage * sum logπ) + β * sum KL.
    KL estimator: exp(r) - 1 - r, with r = logπ - logπ_ref (detached ref).
    """
    if policy_lp.numel() == 0:
        return policy_lp.sum()  # zero scalar on the right device
    pg = -advantage * policy_lp.sum()
    if beta > 0:
        logratio = policy_lp - ref_lp
        kl = torch.exp(logratio) - 1.0 - logratio
        pg = pg + beta * kl.sum()
    return pg


def _chunked(items, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _live_data_objective_no_grad(
    model,
    agent,
    records,
    router_adv,
    solver_adv,
    beta: float,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """Compute the full GRPO objective on real rollout data with no-grad."""
    scale = 1.0 / max(1, len(records))
    total = torch.zeros((), device=device)

    # Router pass
    agent._set_adapter(agent.router_adapter)
    with torch.no_grad():
        for chunk in _chunked(records, chunk_size):
            chunk_sum = torch.tensor(0.0, device=device)
            for _, ro, _, _, _ in chunk:
                policy_lp = batched_teacher_forced_logprobs(
                    model, [ro.router_prompt_ids], [ro.router_completion_ids], device
                )
                ref_lp = batched_reference_logprobs(
                    model, [ro.router_prompt_ids], [ro.router_completion_ids], device
                )
                chunk_sum = chunk_sum + grpo_term(
                    policy_lp[0],
                    ref_lp[0],
                    router_adv[id(ro)],
                    beta,
                )
            total = total + chunk_sum

    # Solver pass
    agent._set_adapter(agent.solver_adapter)
    with torch.no_grad():
        for chunk in _chunked(records, chunk_size):
            chunk_sum = torch.tensor(0.0, device=device)
        for _, ro, _, _, _ in chunk:
            if ro.steps:
                s_adv = solver_adv[id(ro)]
                for si, s in enumerate(ro.steps):
                    s_policy = batched_teacher_forced_logprobs(
                        model, [s.prompt_ids], [s.completion_ids], device
                    )
                    s_ref = batched_reference_logprobs(
                        model, [s.prompt_ids], [s.completion_ids], device
                    )
                    chunk_sum = chunk_sum + grpo_term(
                        s_policy[0],
                        s_ref[0],
                        s_adv,
                        beta,
                    )
            total = total + chunk_sum

    return total * scale


def _run_backward_terms(
    model,
    agent,
    records,
    router_adv,
    solver_adv,
    beta: float,
    chunked: bool,
    chunk_size: int,
    scale: float,
    device: str,
    router_adapter: str,
    solver_adapter: str,
) -> torch.Tensor:
    """
    Compute router+solver GRPO loss terms and run backward.
    Returns a detached total loss value for logging.
    """
    total_loss = torch.zeros((), device=device)

    def chunk_iter():
        if (not chunked) or (chunk_size <= 0) or (chunk_size >= len(records)):
            yield records
        else:
            yield from _chunked(records, chunk_size)

    # Router terms (single adapter context)
    agent._set_adapter(router_adapter)
    model.train()
    for chunk in chunk_iter():
        chunk_sum = torch.zeros((), device=device)
        valid_chunk = [(ro, router_adv[id(ro)]) for _, ro, _, _, _ in chunk]
        if not valid_chunk:
            continue
            
        p_ids = [ro.router_prompt_ids for ro, _ in valid_chunk]
        c_ids = [ro.router_completion_ids for ro, _ in valid_chunk]
        advs = [adv for _, adv in valid_chunk]
        
        r_policies = batched_teacher_forced_logprobs(model, p_ids, c_ids, device)
        r_refs = batched_reference_logprobs(model, p_ids, c_ids, device)
        
        for r_pol, r_ref, adv in zip(r_policies, r_refs, advs):
            r_term = grpo_term(r_pol, r_ref, adv, beta)
            chunk_sum = chunk_sum + r_term
            total_loss = total_loss + r_term.detach()
            
        if chunk_sum.requires_grad:
            (chunk_sum * scale).backward()

    # Solver terms (single adapter context)
    agent._set_adapter(solver_adapter)
    for chunk in chunk_iter():
        chunk_sum = torch.zeros((), device=device)
        valid_steps = []
        for _, ro, _, _, _ in chunk:
            if not ro.steps: continue
            adv = solver_adv[id(ro)]
            for s in ro.steps:
                valid_steps.append((s, adv))
                
        if not valid_steps: continue
        
        # Batching solver passes (up to chunk_size * max_subgoals sequences)
        p_ids = [s.prompt_ids for s, _ in valid_steps]
        c_ids = [s.completion_ids for s, _ in valid_steps]
        advs = [adv for _, adv in valid_steps]
        
        s_policies = batched_teacher_forced_logprobs(model, p_ids, c_ids, device)
        s_refs = batched_reference_logprobs(model, p_ids, c_ids, device)
        
        for s_pol, s_ref, adv in zip(s_policies, s_refs, advs):
            s_term = grpo_term(s_pol, s_ref, adv, beta)
            chunk_sum = chunk_sum + s_term
            total_loss = total_loss + s_term.detach()
            
        if chunk_sum.requires_grad:
            (chunk_sum * scale).backward()

    return total_loss * scale


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def main():
    # Heavy-weight imports kept inside main() so the module can be imported
    # (and helper functions unit-tested) without peft/transformers/datasets.
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    from src.utils.config import load_config
    from src.env.gsm8k_loader import extract_numeric_answer
    from src.agents.router_solver_agent import RouterSolverAgent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/router_solver.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.logging["output_dir"], exist_ok=True)

    # Initialize Weights & Biases
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        # Entity is determined by the logged-in user's API key, not hardcoded
        wandb.init(
            project="router-solver",
            name=f"router-solver-{cfg.training.reward_mode}",
            config={
                "model": cfg.model.base_id,
                "batch_size": cfg.training.batch_size,
                "group_size": cfg.training.group_size,
                "learning_rate": cfg.training.learning_rate,
                "beta": cfg.training.beta,
                "reward_mode": cfg.training.reward_mode,
                "max_steps": cfg.training.max_steps,
            }
        )

    # 1. Model + two LoRA adapters on one frozen base
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa" if device == "cuda" else None,
    )
    lora_cfg = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg, adapter_name=cfg.model.router_adapter_name)
    model.add_adapter(cfg.model.solver_adapter_name, lora_cfg)
    model.to(device)
    model.train()
    enable_gradient_checkpointing = os.getenv(
        "ROUTER_SOLVER_GRADIENT_CHECKPOINTING",
        "1",
    ).lower() in {"1", "true", "yes"}
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[train] gradient checkpointing enabled")
    else:
        print("[train] gradient checkpointing disabled")
    use_compile = os.getenv("ROUTER_SOLVER_TRAIN_COMPILE", "0").lower() in {"1", "true", "yes"}
    if use_compile and device == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed: {e}. Proceeding without compilation.")

    use_vllm = os.getenv("ROUTER_SOLVER_USE_VLLM", "0").lower() in {"1", "true", "yes"}
    vllm_engine = None
    if use_vllm:
        from vllm import LLM
        print("[train] initializing vLLM engine for generation...")
        vllm_engine = LLM(
            model=cfg.model.base_id,
            enable_lora=True,
            max_lora_rank=cfg.model.lora_r,
            max_model_len=4096,
            gpu_memory_utilization=0.45,
            enable_prefix_caching=True,
            enforce_eager=True,
        )

    agent = RouterSolverAgent(model, tokenizer, cfg, device=device)

    # 2. Data
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if os.getenv("ROUTER_SOLVER_SLIM_DATASET", "0") == "1":
        ds = ds.select(range(len(ds) // 8))
        print(f"[train] running on slim dataset, size={len(ds)}")
    ds = ds.shuffle(seed=0)
    questions: List[Tuple[str, int]] = []
    for ex in ds:
        gt = extract_numeric_answer(ex["answer"])
        if gt is not None:
            questions.append((ex["question"], gt))

    # 3. Optional memory (Plan Memory extension, docs/07)
    memory = None
    if cfg.memory.enabled and cfg.memory.mode != "none":
        from src.memory.retrieval import PlanMemory, RetrievalMode
        memory = PlanMemory(
            mode=RetrievalMode(cfg.memory.mode),
            capacity=cfg.memory.capacity,
        )

    # 4. Optimizer — only LoRA parameters have requires_grad
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=cfg.training.learning_rate)

    B = int(os.getenv("ROUTER_SOLVER_BATCH_SIZE", str(cfg.training.batch_size)))
    G = int(os.getenv("ROUTER_SOLVER_GROUP_SIZE", str(cfg.training.group_size)))
    beta = cfg.training.beta
    reward_mode = cfg.training.reward_mode
    loss_chunk_size = int(os.getenv("ROUTER_SOLVER_LOSS_CHUNK_SIZE", "0"))
    lora_synced = False  # Track whether LoRA weights need re-sync to /dev/shm
    max_steps = int(os.getenv("ROUTER_SOLVER_MAX_STEPS", str(cfg.training.max_steps)))
    parity_verify = os.getenv("ROUTER_SOLVER_PARITY_VERIFY", "0").lower() in {"1", "true", "yes"}
    profile_steps = int(os.getenv("ROUTER_SOLVER_PROFILE_STEPS", "0"))
    profile_output_dir = os.getenv(
        "ROUTER_SOLVER_PROFILE_OUTPUT",
        os.path.join(cfg.logging["output_dir"], "profiles"),
    )
    os.makedirs(profile_output_dir, exist_ok=True)
    print(
        f"[train] profiler_steps={profile_steps} profile_output_dir={profile_output_dir}"
    )
    print(f"[train] device={device} B={B} G={G} reward_mode={reward_mode} beta={beta}")

    cursor = 0
    completed_steps = 0
    seen_step_time = 0.0
    pbar = tqdm(range(max_steps), desc="Training", unit="step")
    for step in pbar:
        profile_ctx = None
        if profile_steps > 0 and step < profile_steps:
            from torch.profiler import ProfilerActivity, profile

            profile_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                with_modules=True,
            )
            profile_ctx.__enter__()

        step_start = time()
        tqdm.write(f"[train][step={step}] start")
        # -------------------- Sample batch of B questions --------------------
        batch = []
        for _ in range(B):
            batch.append(questions[cursor % len(questions)])
            cursor += 1

        # -------------------- Rollouts (G per question) ----------------------
        rollout_start = time()
        model.eval()  # deterministic forward (no dropout) during sampling
        records = []  # list of (q_index, Rollout, router_r, step_rs, outcome)
        
        queries = []
        qis = []
        gts = []
        for qi, (q, gt) in enumerate(batch):
            for _ in range(G):
                queries.append(q)
                qis.append(qi)
                gts.append(gt)

        # Use a safe batch size for generation so we don't overwhelm GPU
        gen_batch_size = int(os.getenv("ROUTER_SOLVER_GEN_BATCH_SIZE", "4"))
        
        lora_base_path = None
        if use_vllm:
            lora_base_path = "/dev/shm/router_solver_lora"
            os.makedirs(lora_base_path, exist_ok=True)
            # Only sync LoRA weights when they've changed (after optimizer.step())
            if not lora_synced:
                for adapter in [cfg.model.router_adapter_name, cfg.model.solver_adapter_name]:
                    model.save_pretrained(os.path.join(lora_base_path, adapter), selected_adapters=[adapter])
                lora_synced = True
        
        with torch.no_grad():
            rollouts = agent.batched_rollout(
                queries, memory=memory, do_sample=True, temperature=1.0, batch_size=gen_batch_size,
                vllm_engine=vllm_engine, lora_base_path=lora_base_path
            )
            for qi, ro, gt in zip(qis, rollouts, gts):
                r_router, r_steps, outcome = compute_rewards(ro, gt, reward_mode)
                records.append((qi, ro, r_router, r_steps, outcome))

                # Memory write-gate (docs/07 §Write policy)
                if memory is not None and outcome == 1.0 and ro.plan_dict is not None:
                    memory.write_if_success(
                        ro.question, ro.plan_dict,
                        reward=outcome,
                        tool_errors=ro.tool_error_count,
                    )
        rollout_time = time() - rollout_start
        tqdm.write(
            f"[train][step={step}] collected_rollouts n_records={len(records)} "
            f"rollout_time_sec={rollout_time:.2f}"
        )

        if not records:
            continue

        # -------------------- Group-relative advantages ----------------------
        by_q = defaultdict(list)
        for rec in records:
            by_q[rec[0]].append(rec)

        router_adv = {}   # id(Rollout) → float
        solver_adv = {}
        for qi, group in by_q.items():
            r_r = [r[2] for r in group]
            # Aggregate solver reward per rollout = mean over step rewards (0 if no steps)
            r_s = [sum(r[3]) / len(r[3]) if r[3] else 0.0 for r in group]
            for rec, a_r, a_s in zip(group, group_normalize(r_r), group_normalize(r_s)):
                router_adv[id(rec[1])] = a_r
                solver_adv[id(rec[1])] = a_s

        # -------------------- Loss computation -------------------------------
        total_records = max(1, len(records))
        chunking_requested = loss_chunk_size > 0
        chunk_size = max(1, min(loss_chunk_size, total_records)) if chunking_requested else total_records
        scale = 1.0 / total_records
        if chunking_requested:
            tqdm.write(f"[train] loss chunking requested: chunk_size={chunk_size} total_records={total_records}")
        else:
            tqdm.write("[train] loss chunking disabled")

        if chunking_requested:
            chunk_count = ((total_records + chunk_size - 1) // chunk_size)
            tqdm.write(f"[train] chunked backward will use {chunk_count} chunk(s)")

        optimizer.zero_grad(set_to_none=True)
        try:
            loss = _run_backward_terms(
                model,
                agent,
                records,
                router_adv,
                solver_adv,
                beta,
                chunking_requested,
                chunk_size,
                scale,
                device,
                cfg.model.router_adapter_name,
                cfg.model.solver_adapter_name,
            )
        except torch.cuda.OutOfMemoryError:
            # Fall back to chunked backward only if needed. This preserves baseline
            # path for speed when memory allows, while remaining safe under pressure.
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            fallback_size = min(4, total_records)
            tqdm.write(
                f"[train][oom] full-batch backward OOM, retrying with chunk_size={fallback_size}"
            )
            loss = _run_backward_terms(
                model,
                agent,
                records,
                router_adv,
                solver_adv,
                beta,
                True,
                fallback_size,
                scale,
                device,
                cfg.model.router_adapter_name,
                cfg.model.solver_adapter_name,
            )
        if parity_verify:
            with torch.no_grad():
                was_training = model.training
                model.eval()
                single_chunk = _live_data_objective_no_grad(
                    model,
                    agent,
                    records,
                    router_adv,
                    solver_adv,
                    beta,
                    len(records),
                    device,
                )
                chunked_pass = _live_data_objective_no_grad(
                    model,
                    agent,
                    records,
                    router_adv,
                    solver_adv,
                    beta,
                    chunk_size if chunking_requested else total_records,
                    device,
                )
                parity_gap = (single_chunk - chunked_pass).abs().item()
                tqdm.write(f"[train][parity] live_data_gap={parity_gap:.10f}")
                if was_training:
                    model.train()
        optimizer.step()
        lora_synced = False  # Weights changed, re-sync LoRA on next step
        step_time = time() - step_start
        if profile_ctx is not None:
            profile_ctx.__exit__(None, None, None)
            profile_json = os.path.join(profile_output_dir, f"step_{step}.json")
            profile_ctx.export_chrome_trace(profile_json)
            try:
                tqdm.write(
                    f"[train][step={step}] profile_top_cpu_cuda="
                    + profile_ctx.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=20
                    ).replace("\n", " | ")
                )
            except Exception:
                pass

        seen_step_time += step_time
        completed_steps += 1
        avg_step_time = seen_step_time / completed_steps
        remaining_steps = max_steps - step - 1
        eta_seconds = avg_step_time * max(0, remaining_steps)
        eta_minutes = eta_seconds / 60.0
        eta_hours = eta_seconds / 3600.0
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        tqdm.write(f"[train][step={step}] step_time_sec={step_time:.2f} max_mem_mb={max_mem_mb:.2f}")
        print(
            f"[train][step={step}] throughput_est="
            f"avg_step_sec={avg_step_time:.2f} "
            f"eta_seconds={eta_seconds:.1f} "
            f"eta_minutes={eta_minutes:.2f} "
            f"eta_hours={eta_hours:.2f}"
        )
        pbar.set_postfix(
            {
                "step": step,
                "mem_gb": f"{max_mem_mb/1024:.2f}",
                "eta_h": f"{eta_hours:.2f}",
            }
        )

        # -------------------- Logging ---------------------------------------
        if step % 10 == 0 or step == max_steps - 1:
            outs = [r[4] for r in records]
            rr = [r[2] for r in records]
            sr_means = [sum(r[3]) / len(r[3]) if r[3] else 0.0 for r in records]
            invalid_plans = sum(1 for _, ro, *_ in records if ro.plan_dict is None)

            metrics = {
                "step": step,
                "loss": loss.item(),
                "outcome_acc": np.mean(outs),
                "router_reward": np.mean(rr),
                "solver_reward": np.mean(sr_means),
                "invalid_plans": invalid_plans,
                "total_records": len(records),
            }
            if memory:
                metrics["memory_size"] = len(memory.store.keys)

            tqdm.write(
                f"step={step:4d} "
                f"loss={loss.item():+.4f} "
                f"outcome_acc={np.mean(outs):.3f} "
                f"router_r={np.mean(rr):.3f} "
                f"solver_r={np.mean(sr_means):.3f} "
                f"invalid_plans={invalid_plans}/{len(records)} "
                f"mem={len(memory.store.keys) if memory else 0}"
            )

            if wandb_api_key:
                wandb.log(metrics)

        if step > 0 and step % 50 == 0:
            ckpt = os.path.join(cfg.logging["output_dir"], f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            tqdm.write(f"[train] saved checkpoint: {ckpt}")
            if wandb_api_key:
                wandb.log({"checkpoint_saved": ckpt, "checkpoint_step": step})
    pbar.close()

    # Final save
    final = os.path.join(cfg.logging["output_dir"], "final_hierarchical_model")
    model.save_pretrained(final)
    print(f"[train] done. saved {final}")

    if wandb_api_key:
        wandb.log({
            "final_model_path": final,
            "training_complete": True,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
