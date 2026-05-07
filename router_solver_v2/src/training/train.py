import torch
import numpy as np
import os
import logging
from typing import List, Tuple
from tqdm import tqdm

from src.agents.agent import Agent
from src.rewards.shaper import RewardShaper, Scheduler
from src.utils.config import Config

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


def train_epoch(
    agent: Agent,
    shaper: RewardShaper,
    scheduler: Scheduler,
    questions: List[str],
    ground_truths: List[str],
    optimizer,
    cfg: Config,
    epoch: int,
):
    """Train one epoch using GRPO."""

    r_w, s_w, o_w = scheduler.get_weights(epoch)
    total_loss = 0.0
    correct = 0
    num_questions = 0
    total_rollouts = 0

    for q_idx, (question, gt) in enumerate(zip(questions, ground_truths)):
        # GRPO: Generate G=4 rollouts per question
        rollouts = []
        for _ in range(cfg.rollouts_per_q):
            rollout = agent.rollout(question, gt)
            if rollout.plan and rollout.steps:
                rollouts.append(rollout)

        if not rollouts:
            logger.debug(f"No valid rollouts for question {q_idx+1}")
            continue

        num_questions += 1

        # Batch judge plans for all rollouts in this group
        plan_batch = [(rollout.question, rollout.plan) for rollout in rollouts]
        plan_scores = shaper.judge.batch_judge_plans(plan_batch) if plan_batch else []

        # Batch judge all steps for all rollouts
        step_batch = []
        step_batch_map = {}
        for r_idx, rollout in enumerate(rollouts):
            for step_idx, step in enumerate(rollout.steps):
                step_batch.append((rollout.question, rollout.plan, step_idx, step.reasoning))
                step_batch_map[(r_idx, step_idx)] = len(step_batch) - 1

        step_scores = shaper.judge.batch_judge_steps(step_batch) if step_batch else []

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
            rollout._outcome_reward = shaper.judge.judge_answer("", rollout.final_answer, rollout.ground_truth)

        # Compute rewards for all rollouts in group
        group_rewards = {
            'router': [],
            'steps': [],
            'outcome': [],
        }

        for rollout in rollouts:
            total_rollouts += 1
            group_rewards['router'].append(rollout._router_reward)
            group_rewards['steps'].append(sum(rollout._step_rewards) / len(rollout._step_rewards) if rollout._step_rewards else 0)
            group_rewards['outcome'].append(rollout._outcome_reward)

            # Track accuracy
            if rollout.final_answer and rollout.final_answer.strip().lower() == gt.strip().lower():
                correct += 1

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

            loss = -(
                r_w * router_advantage * router_logp +
                s_w * steps_advantage * sum(step_logps) / len(step_logps) +
                o_w * outcome_advantage * step_logps[-1]
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
            acc = correct / max(total_rollouts, 1)
            print(f"Q {q_idx+1} | loss={avg_loss:.4f} | acc={acc:.1%} ({correct}/{total_rollouts}) | valid_q={num_questions}")
            logger.debug(f"Advantage stats - router: {np.mean(router_advantages):.3f} | steps: {np.mean(steps_advantages):.3f} | outcome: {np.mean(outcome_advantages):.3f}")
            logger.debug(f"Reward stats - router: μ={np.mean(group_rewards['router']):.3f} | steps: μ={np.mean(group_rewards['steps']):.3f}")

        if (q_idx + 1) % cfg.checkpoint_every == 0:
            ckpt_path = f"checkpoint_epoch{epoch}_q{q_idx+1}.pt"
            checkpoint = {
                "epoch": epoch,
                "q_idx": q_idx,
                "model": agent.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_questions": num_questions,
                "total_rollouts": total_rollouts,
                "correct": correct,
                "total_loss": total_loss,
            }
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    final_loss = total_loss / max(num_questions, 1)
    final_acc = correct / max(total_rollouts, 1)

    print(f"\nEpoch summary:")
    print(f"  Questions processed: {num_questions}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  Valid rollout rate: {total_rollouts / max(num_questions, 1):.2f} per question")

    return final_loss, final_acc


def train(
    agent: Agent,
    train_questions: List[str],
    train_gts: List[str],
    cfg: Config,
    optimizer,
):
    """Train for full Phase 4."""

    shaper = RewardShaper(use_judge=cfg.use_judge)
    scheduler = Scheduler(
        router_w=cfg.router_weight,
        solver_w=cfg.solver_weight,
        outcome_w=cfg.outcome_weight,
        decay=cfg.router_weight_decay,
    )

    print(f"\nPhase 4: Full Scale Training")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Rollouts per Q: {cfg.rollouts_per_q}")
    print(f"Total per batch: {cfg.total_per_batch}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Total rollouts: {len(train_questions) * cfg.rollouts_per_q * cfg.epochs:,}")
    print(f"Estimated judge cost: ${len(train_questions) * cfg.rollouts_per_q * cfg.epochs * 3 * 0.0005 / 100:.2f} (batched)")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"LoRA: {cfg.use_lora} (rank={cfg.lora_rank})")
    print()

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        loss, acc = train_epoch(
            agent=agent,
            shaper=shaper,
            scheduler=scheduler,
            questions=train_questions,
            ground_truths=train_gts,
            optimizer=optimizer,
            cfg=cfg,
            epoch=epoch,
        )
        print(f"Epoch {epoch+1} complete: loss={loss:.3f}, acc={acc:.1%}")

    print("\nTraining complete!")
