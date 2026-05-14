import os
import json
from tqdm import tqdm
from typing import List

from src.agent import LinearReasoningAgent
from src.reward import compute_reward


def evaluate(
    agent: LinearReasoningAgent,
    questions: List[str],
    ground_truths: List[str],
    cfg,
    output_path: str = None,
    batch_size: int = 16,
):
    # Evaluate on test set with greedy decoding (batched).
    results = {
        "correct": 0,
        "total": 0,
        "details": [],
    }

    total = len(questions)
    progress = tqdm(range(0, total, batch_size), desc="Evaluating")

    for start in progress:
        end = min(start + batch_size, total)
        q_batch = questions[start:end]
        gt_batch = ground_truths[start:end]

        trajectories = agent.rollout_batch(q_batch, gt_batch, temperature=cfg.eval_temperature)

        for traj in trajectories:
            reward, is_correct, pred = compute_reward(traj.text, traj.ground_truth, cfg)

            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            results["details"].append({
                "question": traj.question[:200],
                "ground_truth": traj.ground_truth,
                "predicted": pred,
                "correct": is_correct,
                "output": traj.text[:400],
            })

        progress.set_postfix({
            "acc": f"{results['correct']}/{results['total']} ({results['correct']/results['total']:.1%})"
        })

    results["accuracy"] = results["correct"] / results["total"]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
