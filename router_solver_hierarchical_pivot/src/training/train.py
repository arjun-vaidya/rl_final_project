import torch
import numpy as np
import os
import logging
import re
from typing import List, Optional
from tqdm import tqdm

from src.agents.agent import Agent
from src.rewards.shaper import RewardShaper, Scheduler
from src.utils.config import Config
from src.utils.answer_utils import clean_answer_text, extract_numeric_value
from src.utils.rollout_trace import append_rollout_trace

logger = logging.getLogger(__name__)


def compute_logp(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, model, tokenizer) -> torch.Tensor:
    """Compute log probability of completion given prompt (prompt tokens ignored). Returns tensor for gradient computation."""
    full_ids = torch.cat([prompt_ids.unsqueeze(0), completion_ids.unsqueeze(0)], dim=1)
    labels = full_ids.clone()
    labels[0, :len(prompt_ids)] = -100
    outputs = model(full_ids, labels=labels)
    return -outputs.loss


def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO: Compute advantages as (reward - mean) / std within group.
    Group size G typically 4.
    """
    if len(rewards) < 2:
        return [0.0] * len(rewards)

    rewards_array = np.array(rewards)

    # Check for NaN/inf
    if np.any(np.isnan(rewards_array)) or np.any(np.isinf(rewards_array)):
        logger.warning(f"Invalid rewards (NaN/inf): {rewards}")
        return [0.0] * len(rewards)

    mean = rewards_array.mean()
    std = rewards_array.std()

    if std < 1e-8:
        logger.debug(f"Rewards have zero variance: {rewards}")
        return [0.0] * len(rewards)

    advantages = (rewards_array - mean) / (std + 1e-8)
    advantages = np.clip(advantages, -2.0, 2.0)
    return advantages.tolist()


def is_exact_match(answer: str, ground_truth: str) -> bool:
    return clean_answer_text(answer).lower() == clean_answer_text(ground_truth).lower()


def is_relaxed_numeric_match(answer: str, ground_truth: str) -> bool:
    answer_num = extract_numeric_value(answer)
    gt_num = extract_numeric_value(ground_truth)
    if answer_num is None or gt_num is None:
        return False
    return abs(answer_num - gt_num) < 1e-6


def _extract_all_numeric_values(text: str) -> List[float]:
    if not text:
        return []
    values: List[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", clean_answer_text(text)):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def train_epoch(
    agent: Agent,
    shaper: RewardShaper,
    scheduler: Scheduler,
    questions: List[str],
    ground_truths: List[str],
    optimizer,
    cfg: Config,
    epoch: int,
    start_q_idx: int = 0,
):
    """Train one epoch using GRPO."""

    r_w, s_w, o_w = scheduler.get_weights(epoch)
    total_loss = 0.0
    relaxed_correct = 0
    exact_correct = 0
    num_questions = 0
    total_rollouts = 0
    kept_groups = 0
    dropped_all_correct = 0
    dropped_all_wrong = 0
    consecutive_dropped = 0
    trace_path = cfg.rollout_trace_path if getattr(cfg, "save_rollout_traces", False) else ""

    print(f"Starting epoch {epoch+1}: {len(questions) - start_q_idx} questions")

    def _outcome_credit_indices(rollout, num_steps: int) -> List[int]:
        if num_steps <= 0:
            return []
        mode = getattr(cfg, "outcome_credit_mode", "last")
        if mode == "all":
            return list(range(num_steps))
        if mode == "answer_bearing":
            answer_idx = rollout.answer_bearing_step_idx
            if answer_idx is None or answer_idx < 0 or answer_idx >= num_steps:
                try:
                    answer_idx = agent._choose_answer_bearing_step_idx(rollout.question, rollout)
                except Exception:
                    answer_idx = None
                rollout.answer_bearing_step_idx = answer_idx
            last_idx = num_steps - 1
            if answer_idx is None:
                return [last_idx]
            start = min(answer_idx, last_idx)
            return list(range(start, last_idx + 1))
        if mode == "answer_bearing_final":
            answer_idx = rollout.answer_bearing_step_idx
            if answer_idx is None or answer_idx < 0 or answer_idx >= num_steps:
                try:
                    answer_idx = agent._choose_answer_bearing_step_idx(rollout.question, rollout)
                except Exception:
                    answer_idx = None
                rollout.answer_bearing_step_idx = answer_idx
            last_idx = num_steps - 1
            if answer_idx is None:
                return [last_idx]
            return sorted({answer_idx, last_idx})
        if mode == "dependency_local":
            answer_idx = rollout.answer_bearing_step_idx
            if answer_idx is None or answer_idx < 0 or answer_idx >= num_steps:
                try:
                    answer_idx = agent._choose_answer_bearing_step_idx(rollout.question, rollout)
                except Exception:
                    answer_idx = None
                rollout.answer_bearing_step_idx = answer_idx
            if answer_idx is None:
                answer_idx = num_steps - 1
            indices = {answer_idx}
            if answer_idx - 1 >= 0:
                indices.add(answer_idx - 1)
            return sorted(indices)
        if mode == "structured_component":
            answer_idx = rollout.answer_bearing_step_idx
            if answer_idx is None or answer_idx < 0 or answer_idx >= num_steps:
                try:
                    answer_idx = agent._choose_answer_bearing_step_idx(rollout.question, rollout)
                except Exception:
                    answer_idx = None
                rollout.answer_bearing_step_idx = answer_idx
            if answer_idx is None:
                answer_idx = num_steps - 1
            indices = {answer_idx}
            if answer_idx - 1 >= 0:
                indices.add(answer_idx - 1)
            return sorted(indices)
        return [num_steps - 1]

    def _answer_bearing_step_correct(rollout) -> bool:
        if not rollout.steps:
            return False
        idx = rollout.answer_bearing_step_idx
        if idx is None or idx < 0 or idx >= len(rollout.steps):
            try:
                idx = agent._choose_answer_bearing_step_idx(rollout.question, rollout)
            except Exception:
                idx = None
            rollout.answer_bearing_step_idx = idx
        if idx is None or idx < 0 or idx >= len(rollout.steps):
            return False
        return is_relaxed_numeric_match(rollout.steps[idx].answer, rollout.ground_truth)

    def _plan_endpoint_answer_like(rollout) -> bool:
        if not rollout.plan:
            return False
        try:
            return bool(agent._is_answer_bearing_subgoal(rollout.plan[-1]))
        except Exception:
            return False

    def _correct_number_in_trace(rollout) -> bool:
        gt_num = extract_numeric_value(rollout.ground_truth)
        if gt_num is None:
            return False
        trace_values: List[float] = []
        for step in rollout.steps:
            trace_values.extend(_extract_all_numeric_values(getattr(step, "answer", "")))
            trace_values.extend(_extract_all_numeric_values(getattr(step, "reasoning", "")))
        return any(abs(value - gt_num) < 1e-6 for value in trace_values)

    for q_idx, (question, gt) in enumerate(zip(questions, ground_truths)):
        if q_idx < start_q_idx:
            continue

        # GRPO: Generate G=4 rollouts per question
        original_solver_max_tokens = agent.solver_max_tokens
        if getattr(cfg, "train_solver_max_tokens", 0):
            agent.solver_max_tokens = cfg.train_solver_max_tokens
        generated_rollouts = agent.rollout_group(question, gt, cfg.rollouts_per_q)
        agent.solver_max_tokens = original_solver_max_tokens
        rollouts = [rollout for rollout in generated_rollouts if rollout.plan and rollout.steps]
        print(f"Q {q_idx+1}: Generated {len(rollouts)}/{cfg.rollouts_per_q} valid rollouts", end=" | ")

        if not rollouts:
            if trace_path:
                exact_matches = [is_exact_match(rollout.final_answer, gt) for rollout in generated_rollouts]
                relaxed_matches = [is_relaxed_numeric_match(rollout.final_answer, gt) for rollout in generated_rollouts]
                append_rollout_trace(
                    trace_path,
                    epoch,
                    q_idx,
                    question,
                    gt,
                    generated_rollouts,
                    exact_matches,
                    relaxed_matches,
                )
            logger.debug(f"No valid rollouts for question {q_idx+1}")
            continue

        num_questions += 1

        judge = shaper.judge

        if judge:
            plan_batch = [(rollout.question, rollout.plan) for rollout in rollouts]
            plan_scores = judge.batch_judge_plans(plan_batch) if plan_batch else []

            step_batch = []
            step_batch_map = {}
            for r_idx, rollout in enumerate(rollouts):
                for step_idx, step in enumerate(rollout.steps):
                    step_batch.append((rollout.question, rollout.plan, step_idx, step.reasoning))
                    step_batch_map[(r_idx, step_idx)] = len(step_batch) - 1

            step_scores = judge.batch_judge_steps(step_batch) if step_batch else []
        else:
            plan_scores = [min(len(rollout.plan) / 8.0, 1.0) for rollout in rollouts]
            step_scores = []
            step_batch_map = {}
            for r_idx, rollout in enumerate(rollouts):
                for step_idx, step in enumerate(rollout.steps):
                    step_batch_map[(r_idx, step_idx)] = len(step_scores)
                    step_scores.append(0.3 if len(step.reasoning) > 50 else 0.1)

        # Assign scores back to rollouts
        total_rollouts_processed = 0
        for r_idx, rollout in enumerate(rollouts):
            total_rollouts_processed += 1

            # Router reward
            router_reward = plan_scores[r_idx] if r_idx < len(plan_scores) else 0.0
            structural = min(len(rollout.plan) / 8.0, 1.0)
            rollout_router_reward = 0.4 * structural + 0.4 * router_reward + 0.2
            rollout._router_reward = min(max(rollout_router_reward, 0.0), 1.0)

            # Step rewards (O(n) instead of O(n²))
            step_rewards = []
            for step_idx in range(len(rollout.steps)):
                score_idx = step_batch_map.get((r_idx, step_idx), -1)
                if score_idx >= 0 and score_idx < len(step_scores):
                    step_rewards.append(step_scores[score_idx])
                else:
                    step_rewards.append(0.0)
            rollout._step_rewards = step_rewards
            if judge:
                rollout._outcome_reward = judge.judge_answer("", rollout.final_answer, rollout.ground_truth)
            else:
                rollout._outcome_reward = shaper._outcome_reward(rollout.final_answer, rollout.ground_truth)

        # Compute rewards for all rollouts in group
        group_rewards = {
            'router': [],
            'steps': [],
            'outcome': [],
        }

        exact_matches = []
        relaxed_matches = []
        answer_bearing_matches = []
        plan_endpoint_flags = []
        correct_number_flags = []
        for rollout in rollouts:
            total_rollouts += 1
            group_rewards['router'].append(rollout._router_reward)
            group_rewards['steps'].append(sum(rollout._step_rewards) / len(rollout._step_rewards) if rollout._step_rewards else 0)
            group_rewards['outcome'].append(rollout._outcome_reward)

            # Track accuracy with relaxed numeric correctness as the primary metric.
            exact_match = is_exact_match(rollout.final_answer, gt)
            relaxed_match = is_relaxed_numeric_match(rollout.final_answer, gt)
            exact_matches.append(exact_match)
            relaxed_matches.append(relaxed_match)
            if exact_match:
                exact_correct += 1
            if relaxed_match:
                relaxed_correct += 1
            answer_bearing_matches.append(_answer_bearing_step_correct(rollout))
            plan_endpoint_flags.append(_plan_endpoint_answer_like(rollout))
            correct_number_flags.append(_correct_number_in_trace(rollout))

        if getattr(cfg, "informative_group_sampling", False):
            n_correct = sum(1 for matched in relaxed_matches if matched)
            final_varies = 0 < n_correct < len(rollouts)
            ab_correct_count = sum(1 for matched in answer_bearing_matches if matched)
            ab_varies = 0 < ab_correct_count < len(answer_bearing_matches) if answer_bearing_matches else False
            plan_endpoint_count = sum(1 for matched in plan_endpoint_flags if matched)
            plan_varies = 0 < plan_endpoint_count < len(plan_endpoint_flags) if plan_endpoint_flags else False
            trace_correct_count = sum(1 for matched in correct_number_flags if matched)
            trace_varies = 0 < trace_correct_count < len(correct_number_flags) if correct_number_flags else False
            informative_mode = getattr(cfg, "informative_group_mode", "final_only")
            keep_group = final_varies or (
                informative_mode == "structured" and (ab_varies or plan_varies or trace_varies)
            )
            if not keep_group:
                if n_correct == 0:
                    dropped_all_wrong += 1
                    consecutive_dropped += 1
                    print(f"drop_all_wrong={dropped_all_wrong}", end=" | ")
                    if consecutive_dropped >= max(getattr(cfg, "informative_max_resamples", 0), 1):
                        logger.warning(
                            "Dropped %d consecutive non-informative groups (all wrong finals) by question %d",
                            consecutive_dropped,
                            q_idx + 1,
                        )
                elif n_correct == len(rollouts):
                    dropped_all_correct += 1
                    consecutive_dropped += 1
                    print(f"drop_all_correct={dropped_all_correct}", end=" | ")
                    if consecutive_dropped >= max(getattr(cfg, "informative_max_resamples", 0), 1):
                        logger.warning(
                            "Dropped %d consecutive non-informative groups (all correct finals) by question %d",
                            consecutive_dropped,
                            q_idx + 1,
                        )
                else:
                    consecutive_dropped += 1
                    print("drop_structured_flat=1", end=" | ")
                continue
            kept_groups += 1
            consecutive_dropped = 0
        else:
            kept_groups += 1
            consecutive_dropped = 0

        if trace_path:
            all_exact_matches = [is_exact_match(rollout.final_answer, gt) for rollout in generated_rollouts]
            all_relaxed_matches = [is_relaxed_numeric_match(rollout.final_answer, gt) for rollout in generated_rollouts]
            append_rollout_trace(
                trace_path,
                epoch,
                q_idx,
                question,
                gt,
                generated_rollouts,
                all_exact_matches,
                all_relaxed_matches,
            )

        # GRPO: Compute advantages within group
        if getattr(cfg, "outcome_credit_mode", "last") == "structured_component":
            router_component_rewards = []
            answer_component_rewards = []
            final_component_rewards = []
            for rollout_idx, rollout in enumerate(rollouts):
                plan_ok = float(plan_endpoint_flags[rollout_idx])
                answer_ok = float(answer_bearing_matches[rollout_idx])
                trace_ok = float(correct_number_flags[rollout_idx])
                final_ok = float(relaxed_matches[rollout_idx])
                router_component_rewards.append(0.5 * plan_ok + 0.5 * max(answer_ok, trace_ok))
                answer_component_rewards.append(1.0 if answer_ok > 0 else (0.5 if trace_ok > 0 else 0.0))
                final_component_rewards.append(1.0 if final_ok > 0 else (-0.5 if trace_ok > 0 else 0.0))
            router_advantages = compute_grpo_advantages(router_component_rewards)
            steps_advantages = compute_grpo_advantages(answer_component_rewards)
            outcome_advantages = compute_grpo_advantages(final_component_rewards)
        else:
            router_advantages = compute_grpo_advantages(group_rewards['router'])
            steps_advantages = compute_grpo_advantages(group_rewards['steps'])
            outcome_advantages = compute_grpo_advantages(group_rewards['outcome'])

        # Update on each rollout with its advantage
        for rollout_idx, rollout in enumerate(rollouts):
            # Compute log probs with gradient tracking
            # IMPORTANT: Set correct adapter for each component to ensure gradients flow to right LoRA weights

            agent._set_adapter("router")
            router_logp = compute_logp(
                rollout.router_prompt_ids,
                rollout.router_completion_ids,
                agent.model,
                agent.tokenizer,
            )

            agent._set_adapter("solver")
            num_steps = len(rollout.steps)
            if num_steps == 0:
                continue

            # GRPO loss: advantage * log_prob
            router_advantage = router_advantages[rollout_idx]
            steps_advantage = steps_advantages[rollout_idx]
            outcome_advantage = outcome_advantages[rollout_idx]
            outcome_credit_indices = set(_outcome_credit_indices(rollout, num_steps))
            outcome_credit_scale = max(len(outcome_credit_indices), 1)
            router_loss = -(r_w * router_advantage * router_logp)
            router_loss_item = router_loss.detach().item()
            if np.isnan(router_loss_item) or np.isinf(router_loss_item):
                logger.warning(f"NaN/Inf router loss at q_idx {q_idx}, rollout {rollout_idx}: {router_loss_item}")
                continue
            router_loss.backward()
            total_loss += router_loss_item

            for step_idx, step in enumerate(rollout.steps):
                step_logp = compute_logp(step.prompt_ids, step.completion_ids, agent.model, agent.tokenizer)
                if getattr(cfg, "outcome_credit_mode", "last") == "structured_component":
                    coeff = 0.0
                    if step_idx in outcome_credit_indices:
                        coeff += s_w * steps_advantage / outcome_credit_scale
                    if step_idx == num_steps - 1:
                        coeff += o_w * outcome_advantage
                else:
                    coeff = s_w * steps_advantage / num_steps
                    if step_idx in outcome_credit_indices:
                        coeff += o_w * outcome_advantage / outcome_credit_scale

                step_loss = -(coeff * step_logp)
                step_loss_item = step_loss.detach().item()
                if np.isnan(step_loss_item) or np.isinf(step_loss_item):
                    logger.warning(f"NaN/Inf step loss at q_idx {q_idx}, rollout {rollout_idx}, step {step_idx}: {step_loss_item}")
                    continue
                step_loss.backward()
                total_loss += step_loss_item

        optimizer.step()
        optimizer.zero_grad()

        if (q_idx + 1) % cfg.log_every == 0:
            avg_loss = total_loss / max(num_questions, 1)
            relaxed_acc = relaxed_correct / max(total_rollouts, 1)
            exact_acc = exact_correct / max(total_rollouts, 1)
            print(f"PROGRESS | loss={avg_loss:.4f} | acc={relaxed_acc:.1%} ({relaxed_correct}/{total_rollouts}) | valid_q={num_questions}")
            print(f"  Exact acc: {exact_acc:.1%} ({exact_correct}/{total_rollouts})")
            print(f"  Advantages: router={np.mean(router_advantages):.3f}, steps={np.mean(steps_advantages):.3f}, outcome={np.mean(outcome_advantages):.3f}")
            print(f"  Rewards: router={np.mean(group_rewards['router']):.3f}, steps={np.mean(group_rewards['steps']):.3f}")
            print(f"  Informative groups: kept={kept_groups} drop_all_correct={dropped_all_correct} drop_all_wrong={dropped_all_wrong}")
            print(
                f"  Component diag: plan_endpoint_answer_like={sum(plan_endpoint_flags)}/{len(plan_endpoint_flags)} "
                f"answer_bearing_correct={sum(answer_bearing_matches)}/{len(answer_bearing_matches)} "
                f"correct_number_in_trace={sum(correct_number_flags)}/{len(correct_number_flags)} "
                f"final_relaxed_correct={sum(relaxed_matches)}/{len(relaxed_matches)}"
            )

        if (q_idx + 1) % cfg.checkpoint_every == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}_q{q_idx+1}.pt")
            checkpoint = {
                "epoch": epoch,
                "q_idx": q_idx,
                "model": agent.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_questions": num_questions,
                "total_rollouts": total_rollouts,
                "relaxed_correct": relaxed_correct,
                "exact_correct": exact_correct,
                "total_loss": total_loss,
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    final_loss = total_loss / max(num_questions, 1)
    final_relaxed_acc = relaxed_correct / max(total_rollouts, 1)
    final_exact_acc = exact_correct / max(total_rollouts, 1)

    print(f"\nEpoch summary:")
    print(f"  Questions processed: {num_questions}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final relaxed accuracy: {final_relaxed_acc:.1%}")
    print(f"  Final exact accuracy: {final_exact_acc:.1%}")
    print(f"  Valid rollout rate: {total_rollouts / max(num_questions, 1):.2f} per question")
    print(f"  Informative groups: kept={kept_groups} drop_all_correct={dropped_all_correct} drop_all_wrong={dropped_all_wrong}")

    return final_loss, final_relaxed_acc


def train(
    agent: Agent,
    train_questions: List[str],
    train_gts: List[str],
    cfg: Config,
    optimizer,
    start_epoch: int = 0,
    start_q_idx: int = 0,
):
    """Train for full Phase 4."""

    shaper = RewardShaper(use_judge=cfg.use_judge)
    scheduler = Scheduler(
        router_w=cfg.router_weight,
        solver_w=cfg.solver_weight,
        outcome_w=cfg.outcome_weight,
        decay=cfg.router_weight_decay,
    )

    print(f"\n" + "="*70)
    print(f"PHASE 4: FULL SCALE TRAINING")
    print(f"="*70)
    print(f"Questions: {len(train_questions)}")
    print(f"Rollouts per Q: {cfg.rollouts_per_q}")
    print(f"Total rollouts: {len(train_questions) * cfg.rollouts_per_q * cfg.epochs:,}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"LoRA adapters: default, router, solver (rank={cfg.lora_rank})")
    print(f"="*70 + "\n")

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        epoch_start_q_idx = start_q_idx if epoch == start_epoch else 0
        loss, acc = train_epoch(
            agent=agent,
            shaper=shaper,
            scheduler=scheduler,
            questions=train_questions,
            ground_truths=train_gts,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
            start_q_idx=epoch_start_q_idx,
        )
        print(f"Epoch {epoch+1} complete: loss={loss:.3f}, acc={acc:.1%}")

    print("\nTraining complete!")
