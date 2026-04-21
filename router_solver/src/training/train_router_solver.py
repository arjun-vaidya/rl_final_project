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

import numpy as np
import torch
import torch.nn.functional as F

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
    else:
        r_router = router_reward(rollout.router_output, final_ans, gt)
        r_steps = [solver_step_reward(s.tool_result, outcome) for s in rollout.steps]

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

def teacher_forced_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Return per-token log-probs of `completion_ids` given `prompt_ids`. Grad flows."""
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    if completion_ids.numel() == 0:
        return torch.zeros(0, device=device)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)  # [1, L]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]                                               # [L, V]

    prompt_len = prompt_ids.size(0)
    comp_len = completion_ids.size(0)

    # Logits at positions [prompt_len-1, prompt_len+comp_len-1) predict the
    # completion tokens at [prompt_len, prompt_len+comp_len).
    shift_logits = logits[prompt_len - 1: prompt_len - 1 + comp_len, :]      # [L_c, V]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)     # [L_c]


def reference_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Same as above but with all LoRA adapters disabled (= frozen base model).
    No grad, since the reference policy is not being trained."""
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                return teacher_forced_logprobs(model, prompt_ids, completion_ids, device).detach()
        return teacher_forced_logprobs(model, prompt_ids, completion_ids, device).detach()


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

    # 1. Model + two LoRA adapters on one frozen base
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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

    agent = RouterSolverAgent(model, tokenizer, cfg, device=device)

    # 2. Data
    ds = load_dataset("openai/gsm8k", "main", split="train")
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

    B = cfg.training.batch_size
    G = cfg.training.group_size
    beta = cfg.training.beta
    reward_mode = cfg.training.reward_mode
    print(f"[train] device={device} B={B} G={G} reward_mode={reward_mode} beta={beta}")

    cursor = 0
    for step in range(cfg.training.max_steps):
        # -------------------- Sample batch of B questions --------------------
        batch = []
        for _ in range(B):
            batch.append(questions[cursor % len(questions)])
            cursor += 1

        # -------------------- Rollouts (G per question) ----------------------
        model.eval()  # deterministic forward (no dropout) during sampling
        records = []  # list of (q_index, Rollout, router_r, step_rs, outcome)
        for qi, (q, gt) in enumerate(batch):
            for _ in range(G):
                ro = agent.rollout(q, memory=memory, do_sample=True, temperature=1.0)
                r_router, r_steps, outcome = compute_rewards(ro, gt, reward_mode)
                records.append((qi, ro, r_router, r_steps, outcome))

                # Memory write-gate (docs/07 §Write policy)
                if memory is not None and outcome == 1.0 and ro.plan_dict is not None:
                    memory.write_if_success(
                        q, ro.plan_dict,
                        reward=outcome,
                        tool_errors=ro.tool_error_count,
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
        optimizer.zero_grad()
        total_router_loss = torch.zeros((), device=device)
        total_solver_loss = torch.zeros((), device=device)

        for _qi, ro, _rr, _rs, _out in records:
            # Router term ── use router adapter for the policy forward pass
            agent._set_adapter(cfg.model.router_adapter_name)
            r_policy = teacher_forced_logprobs(
                model, ro.router_prompt_ids, ro.router_completion_ids, device
            )
            r_ref = reference_logprobs(
                model, ro.router_prompt_ids, ro.router_completion_ids, device
            )
            total_router_loss = total_router_loss + grpo_term(
                r_policy, r_ref, router_adv[id(ro)], beta
            )

            # Solver term ── sum over subgoals under solver adapter
            if ro.steps:
                agent._set_adapter(cfg.model.solver_adapter_name)
                s_adv = solver_adv[id(ro)]
                for s in ro.steps:
                    s_policy = teacher_forced_logprobs(
                        model, s.prompt_ids, s.completion_ids, device
                    )
                    s_ref = reference_logprobs(
                        model, s.prompt_ids, s.completion_ids, device
                    )
                    total_solver_loss = total_solver_loss + grpo_term(
                        s_policy, s_ref, s_adv, beta
                    )

        # Normalize by rollout count so LR is invariant to batch size
        loss = (total_router_loss + total_solver_loss) / len(records)
        loss.backward()
        optimizer.step()

        # -------------------- Logging ---------------------------------------
        if step % 10 == 0 or step == cfg.training.max_steps - 1:
            outs = [r[4] for r in records]
            rr = [r[2] for r in records]
            sr_means = [sum(r[3]) / len(r[3]) if r[3] else 0.0 for r in records]
            invalid_plans = sum(1 for _, ro, *_ in records if ro.plan_dict is None)
            print(
                f"step={step:4d} "
                f"loss={loss.item():+.4f} "
                f"outcome_acc={np.mean(outs):.3f} "
                f"router_r={np.mean(rr):.3f} "
                f"solver_r={np.mean(sr_means):.3f} "
                f"invalid_plans={invalid_plans}/{len(records)} "
                f"mem={len(memory.store.keys) if memory else 0}"
            )

        if step > 0 and step % 50 == 0:
            ckpt = os.path.join(cfg.logging["output_dir"], f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            print(f"[train] saved checkpoint: {ckpt}")

    # Final save
    final = os.path.join(cfg.logging["output_dir"], "final_hierarchical_model")
    model.save_pretrained(final)
    print(f"[train] done. saved {final}")


if __name__ == "__main__":
    main()
