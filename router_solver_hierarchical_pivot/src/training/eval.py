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
    # Evaluate on a dataset.

    shaper = RewardShaper(use_judge=cfg.use_judge)
    results = {
        "correct": 0,
        "exact_correct": 0,
        "strict_numeric_correct": 0,
        "relaxed_numeric_correct": 0,
        "total": 0,
        "router_rewards": [],
        "step_rewards": [],
        "outcome_rewards": [],
        "plan_validity": 0,
    }

    for q_idx, (question, gt) in enumerate(tqdm(zip(questions, ground_truths), total=len(questions))):
        rollout = agent.rollout(question, gt)
        results["total"] += 1

        if not rollout.is_valid():
            continue

        results["plan_validity"] += 1
        rollout_batch.append(rollout)

        plan_batch.append((rollout.question, rollout.plan))
        for step_idx, step in enumerate(rollout.steps):
            step_batch.append((rollout.question, rollout.plan, step_idx, step.reasoning))

        is_last = (q_idx == len(questions) - 1)
        if len(plan_batch) >= 10 or is_last:
            process_batch(plan_batch, step_batch, rollout_batch)
            plan_batch = []
            step_batch = []
            rollout_batch = []

    # Compute averages
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["exact_accuracy"] = results["exact_correct"] / results["total"] if results["total"] > 0 else 0
    results["strict_numeric_accuracy"] = results["strict_numeric_correct"] / results["total"] if results["total"] > 0 else 0
    results["relaxed_numeric_accuracy"] = results["relaxed_numeric_correct"] / results["total"] if results["total"] > 0 else 0
    results["plan_validity_rate"] = results["plan_validity"] / results["total"] if results["total"] > 0 else 0
    results["avg_router_reward"] = sum(results["router_rewards"]) / len(results["router_rewards"]) if results["router_rewards"] else 0
    results["avg_step_reward"] = sum(results["step_rewards"]) / len(results["step_rewards"]) if results["step_rewards"] else 0
    results["avg_outcome_reward"] = sum(results["outcome_rewards"]) / len(results["outcome_rewards"]) if results["outcome_rewards"] else 0

    return results


def print_eval_results(results: Dict, name: str = ""):
    # Print evaluation results.
    print(f"\n{'='*60}")
    print(f"Evaluation Results {name}")
    print(f"{'='*60}")
    print(f"Accuracy:              {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"Exact accuracy:        {results['exact_accuracy']:.1%} ({results['exact_correct']}/{results['total']})")
    print(f"Strict numeric acc:    {results['strict_numeric_accuracy']:.1%} ({results['strict_numeric_correct']}/{results['total']})")
    print(f"Relaxed numeric acc:   {results['relaxed_numeric_accuracy']:.1%} ({results['relaxed_numeric_correct']}/{results['total']})")
    print(f"Plan validity:         {results['plan_validity_rate']:.1%}")
    print(f"Avg router reward:     {results['avg_router_reward']:.3f}")
    print(f"Avg step reward:       {results['avg_step_reward']:.3f}")
    print(f"Avg outcome reward:    {results['avg_outcome_reward']:.3f}")
    print(f"{'='*60}")
