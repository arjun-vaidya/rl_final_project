import torch
from typing import List, Dict
from tqdm import tqdm

from src.agents.agent import Agent
from src.rewards.shaper import RewardShaper
from src.utils.config import Config


def evaluate(
    agent: Agent,
    questions: List[str],
    ground_truths: List[str],
    cfg: Config,
) -> Dict:
    """Evaluate on a dataset."""

    shaper = RewardShaper(use_judge=cfg.use_judge)
    results = {
        "correct": 0,
        "total": 0,
        "router_rewards": [],
        "step_rewards": [],
        "outcome_rewards": [],
        "plan_validity": 0,
    }

    # Batch judgments for efficiency
    plan_batch = []
    step_batch = []
    rollout_batch = []

    def process_batch(plan_batch, step_batch, rollout_batch):
        """Judge a batch and assign scores to rollouts."""
        if not plan_batch:
            return

        plan_scores = shaper.judge.batch_judge_plans(plan_batch)
        step_scores = shaper.judge.batch_judge_steps(step_batch)

        # Map step scores to rollouts
        step_idx = 0
        for rollout_idx, rollout in enumerate(rollout_batch):
            router_reward = plan_scores[rollout_idx] if rollout_idx < len(plan_scores) else 0.0
            results["router_rewards"].append(router_reward)

            for _ in rollout.steps:
                if step_idx < len(step_scores):
                    results["step_rewards"].append(step_scores[step_idx])
                    step_idx += 1

            outcome = shaper.judge.judge_answer("", rollout.final_answer, rollout.ground_truth)
            results["outcome_rewards"].append(outcome)

            # Check correctness
            if rollout.final_answer and rollout.final_answer.strip().lower() == rollout.ground_truth.strip().lower():
                results["correct"] += 1
            else:
                try:
                    if abs(float(rollout.final_answer) - float(rollout.ground_truth)) < 1e-6:
                        results["correct"] += 1
                except:
                    pass

    for q_idx, (question, gt) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        rollout = agent.rollout(question, gt)
        results["total"] += 1

        if not rollout.is_valid():
            continue

        results["plan_validity"] += 1
        rollout_batch.append(rollout)

        # Collect for batching
        plan_batch.append((rollout.question, rollout.plan))
        for step_idx, step in enumerate(rollout.steps):
            step_batch.append((rollout.question, rollout.plan, step_idx, step.reasoning))

        # Judge in batches of 10, or at end
        is_last = (q_idx == len(questions) - 1)
        if len(plan_batch) >= 10 or is_last:
            process_batch(plan_batch, step_batch, rollout_batch)
            plan_batch = []
            step_batch = []
            rollout_batch = []

    # Compute averages
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["plan_validity_rate"] = results["plan_validity"] / results["total"] if results["total"] > 0 else 0
    results["avg_router_reward"] = sum(results["router_rewards"]) / len(results["router_rewards"]) if results["router_rewards"] else 0
    results["avg_step_reward"] = sum(results["step_rewards"]) / len(results["step_rewards"]) if results["step_rewards"] else 0
    results["avg_outcome_reward"] = sum(results["outcome_rewards"]) / len(results["outcome_rewards"]) if results["outcome_rewards"] else 0

    return results


def print_eval_results(results: Dict, name: str = ""):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results {name}")
    print(f"{'='*60}")
    print(f"Accuracy:              {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"Plan validity:         {results['plan_validity_rate']:.1%}")
    print(f"Avg router reward:     {results['avg_router_reward']:.3f}")
    print(f"Avg step reward:       {results['avg_step_reward']:.3f}")
    print(f"Avg outcome reward:    {results['avg_outcome_reward']:.3f}")
    print(f"{'='*60}")
