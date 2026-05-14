import torch
import numpy as np
import os
import logging
from typing import List, Optional
from tqdm import tqdm

from src.agents.agent import Agent
from src.rewards.shaper import RewardShaper, Scheduler
from src.utils.config import Config
from src.utils.answer_utils import clean_answer_text, extract_numeric_value
from src.utils.rollout_trace import append_rollout_trace

logger = logging.getLogger(__name__)


def compute_logp(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, model, tokenizer) -> torch.Tensor:
    # Compute log probability of completion given prompt (prompt tokens ignored). Returns tensor for gradient computation.
    full_ids = torch.cat([prompt_ids.unsqueeze(0), completion_ids.unsqueeze(0)], dim=1)
    labels = full_ids.clone()
    labels[0, :len(prompt_ids)] = -100
    outputs = model(full_ids, labels=labels)
    return -outputs.loss


def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    # GRPO: Compute advantages as (reward - mean) / std within group.
    # Group size G typically 4.
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


def aggregate_outcome_logp(rollout, step_logps: List[torch.Tensor], use_all_steps: bool, model, tokenizer) -> Optional[torch.Tensor]:
    if not step_logps:
        return None

    outcome_terms = []
    if use_all_steps:
        outcome_terms.extend(step_logps)
    else:
        outcome_terms.append(step_logps[-1])

    if not outcome_terms:
        return None
    return sum(outcome_terms) / len(outcome_terms)


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
    # Train one epoch using GRPO.

    r_w, s_w, o_w = scheduler.get_weights(epoch)
    total_loss = 0.0
    relaxed_correct = 0
    exact_correct = 0
    num_questions = 0
    total_rollouts = 0
    trace_path = cfg.rollout_trace_path if getattr(cfg, "save_rollout_traces", False) else ""

    print(f"Starting epoch {epoch+1}: {len(questions) - start_q_idx} questions")

    for q_idx, (question, gt) in enumerate(zip(questions, ground_truths)):
        if q_idx < start_q_idx:
            continue

        # GRPO: Generate G=4 rollouts per question
        generated_rollouts = agent.rollout_group(question, gt, cfg.rollouts_per_q)
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
            step_logps = [
                compute_logp(s.prompt_ids, s.completion_ids, agent.model, agent.tokenizer)
                for s in rollout.steps
            ]

            if not step_logps:
                continue

            # GRPO loss: advantage * log_prob
            router_advantage = router_advantages[rollout_idx]
            steps_advantage = steps_advantages[rollout_idx]
            outcome_advantage = outcome_advantages[rollout_idx]
            outcome_logp = aggregate_outcome_logp(
                rollout,
                step_logps,
                cfg.outcome_credit_all_steps,
                agent.model,
                agent.tokenizer,
            )
            if outcome_logp is None:
                continue

            loss = -(
                r_w * router_advantage * router_logp +
                s_w * steps_advantage * sum(step_logps) / len(step_logps) +
                o_w * outcome_advantage * outcome_logp
            )

            loss_item = loss.detach().item()
            if np.isnan(loss_item) or np.isinf(loss_item):
                logger.warning(f"NaN/Inf loss at q_idx {q_idx}, rollout {rollout_idx}: {loss_item}")
                continue

            loss.backward()
            total_loss += loss_item

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
    # Train for full Phase 4.

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
