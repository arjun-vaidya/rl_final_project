import os
import json
from collections import Counter
from tqdm import tqdm
from typing import List

from src.agent import LinearReasoningAgent
from src.reward import compute_reward, extract_boxed_answer, numeric_match, extract_last_number


def _vote(predictions: List[str], ground_truth: str):
    """Majority-vote over a list of predicted strings; return (vote_pred, vote_correct, agreement)."""
    normalized = []
    for p in predictions:
        if p is None:
            continue
        try:
            normalized.append(str(float(str(p).replace(",", "").strip())))
        except (ValueError, TypeError):
            f = extract_last_number(str(p))
            if f is not None:
                normalized.append(str(float(f)))
            else:
                normalized.append(str(p))

    if not normalized:
        return None, False, 0.0

    counts = Counter(normalized)
    vote_pred, vote_count = counts.most_common(1)[0]
    agreement = vote_count / len(predictions)
    vote_correct = numeric_match(vote_pred, ground_truth)
    return vote_pred, vote_correct, agreement


def evaluate_sc(
    agent: LinearReasoningAgent,
    questions: List[str],
    ground_truths: List[str],
    cfg,
    K: int = 8,
    temperature: float = 0.6,
    output_path: str = None,
    batch_size: int = 8,
    vllm_engine=None,
):
    """Self-consistency eval: K rollouts per question at `temperature`, majority-vote over \\boxed{} answers.

    Also reports greedy (T=0) accuracy as a control by taking the first rollout of each group
    only if `temperature` is 0; otherwise we report majority-vote accuracy only.
    """
    results = {
        "K": K,
        "temperature": temperature,
        "correct_majority": 0,
        "correct_any": 0,
        "total": 0,
        "mean_agreement": 0.0,
        "details": [],
    }

    progress = tqdm(range(0, len(questions), batch_size), desc=f"Eval SC K={K} T={temperature}")
    agreements = []

    for start in progress:
        end = min(start + batch_size, len(questions))
        q_batch = questions[start:end]
        gt_batch = ground_truths[start:end]

        if vllm_engine is not None:
            # vLLM path: returns K texts per question directly
            texts_per_q = vllm_engine.generate_flat(agent, q_batch, K, temperature)
        else:
            # HF path: one call with B*K prompts, split afterwards
            prompts_flat = []
            for q in q_batch:
                prompts_flat.extend([agent._build_prompt(q)] * K)
            gen = agent._generate_batch(prompts_flat, agent.max_cot_tokens, temperature)
            texts_per_q = [
                [gen[qi * K + ki][0] for ki in range(K)]
                for qi in range(len(q_batch))
            ]

        for qi, (q, gt) in enumerate(zip(q_batch, gt_batch)):
            preds = []
            outputs = []
            any_correct = False
            for ki in range(K):
                text = texts_per_q[qi][ki]
                outputs.append(text)
                r, c, p = compute_reward(text, gt, cfg)
                preds.append(p)
                if c:
                    any_correct = True

            vote_pred, vote_correct, agreement = _vote(preds, gt)
            agreements.append(agreement)

            results["total"] += 1
            if vote_correct:
                results["correct_majority"] += 1
            if any_correct:
                results["correct_any"] += 1

            results["details"].append({
                "question": q[:200],
                "ground_truth": gt,
                "predicted_majority": vote_pred,
                "agreement": round(agreement, 3),
                "majority_correct": vote_correct,
                "any_correct": any_correct,
                "predictions": [str(p) for p in preds],
            })

        progress.set_postfix({
            "maj": f"{results['correct_majority']}/{results['total']} ({results['correct_majority']/max(1,results['total']):.2%})",
            "any": f"{results['correct_any']}/{results['total']} ({results['correct_any']/max(1,results['total']):.2%})",
        })

    results["accuracy_majority"] = results["correct_majority"] / results["total"] if results["total"] else 0.0
    results["accuracy_any"] = results["correct_any"] / results["total"] if results["total"] else 0.0
    results["mean_agreement"] = sum(agreements) / len(agreements) if agreements else 0.0

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
