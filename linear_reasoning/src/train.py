import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List

from src.agent import LinearReasoningAgent
from src.reward import compute_reward


def compute_logprobs_batch(
    model,
    prompt_ids_list: List[torch.Tensor],
    completion_ids_list: List[torch.Tensor],
    pad_token_id: int,
) -> List[torch.Tensor]:
    """
    Batched forward pass. One forward over all (prompt, completion) pairs in the
    microbatch, then per-trajectory extraction of completion-token logprobs.

    Right-pads sequences to the max length in the batch and uses attention_mask to
    prevent padded positions from contributing to attention. Returns a list of
    completion-token logprob tensors (variable length, one per trajectory).
    """
    device = model.device
    B = len(prompt_ids_list)

    sequences = [torch.cat([p, c]) for p, c in zip(prompt_ids_list, completion_ids_list)]
    prompt_lens = [p.size(0) for p in prompt_ids_list]
    completion_lens = [c.size(0) for c in completion_ids_list]
    max_len = max(s.size(0) for s in sequences)

    batch_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        batch_ids[i, : seq.size(0)] = seq.to(device)
        attention_mask[i, : seq.size(0)] = 1

    outputs = model(batch_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    results = []
    for i in range(B):
        p_len = prompt_lens[i]
        c_len = completion_lens[i]
        # logits at position p_len-1 .. p_len-1+c_len-1 predict tokens at p_len .. p_len+c_len-1
        completion_logits = logits[i, p_len - 1 : p_len - 1 + c_len]
        completion_targets = completion_ids_list[i].to(device)
        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze(1)

        # Defensive padding mask. trim_at_eos in agent.py should already truncate at the
        # first EOS, but the model may terminate on a different token (e.g. Qwen's
        # <|im_end|>) and leave EOS padding behind. Mask everything strictly AFTER the
        # first pad/eos occurrence so the model isn't trained on synthetic EOS runs.
        # The first EOS itself stays unmasked — it's a legitimate stop prediction.
        eos_positions = (completion_targets == pad_token_id).nonzero(as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            first_eos = eos_positions[0].item()
            mask = torch.ones_like(token_log_probs)
            mask[first_eos + 1 :] = 0.0
            token_log_probs = token_log_probs * mask

        results.append(token_log_probs)

    return results


@torch.no_grad()
def compute_ref_logprobs_batch(
    model,
    prompt_ids_list: List[torch.Tensor],
    completion_ids_list: List[torch.Tensor],
    pad_token_id: int,
) -> List[torch.Tensor]:
    """Batched reference logprobs with LoRA disabled (no grad)."""
    with model.disable_adapter():
        return [lp.detach() for lp in compute_logprobs_batch(
            model, prompt_ids_list, completion_ids_list, pad_token_id
        )]


def compute_kl_k3(log_p: torch.Tensor, log_p_ref: torch.Tensor) -> torch.Tensor:
    """
    Unbiased k3 KL estimator: KL(p || p_ref) ≈ exp(log_p_ref - log_p) - (log_p_ref - log_p) - 1
    Always non-negative. This is the estimator used in DeepSeek-R1.
    """
    log_ratio = log_p_ref - log_p
    return torch.exp(log_ratio) - log_ratio - 1.0


def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """Compute group-relative advantages: (r - mean) / std."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std = rewards_t.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    advantages = (rewards_t - mean) / (std + 1e-8)
    return advantages.tolist()


def train(
    agent: LinearReasoningAgent,
    questions: List[str],
    ground_truths: List[str],
    cfg,
    optimizer,
):
    """GRPO training loop with verifiable rewards."""
    model = agent.model
    G = cfg.rollouts_per_q
    pad_token_id = agent.tokenizer.pad_token_id
    microbatch_size = max(1, cfg.train_microbatch_size)

    stats = {
        "total_correct": 0,
        "total_rollouts": 0,
        "running_reward": [],
        "running_acc": [],
        "step_losses": [],
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Starting GRPO training: {len(questions)} questions, G={G} rollouts each")
    log(f"Total rollouts: {len(questions) * G}")

    progress = tqdm(zip(questions, ground_truths), total=len(questions), desc="Training")

    for q_idx, (question, gt) in enumerate(progress):
        # Generate G rollouts
        model.eval()
        trajectories = agent.rollout_group(question, gt, G, temperature=cfg.temperature)

        # Score each rollout
        for traj in trajectories:
            reward, is_correct, pred = compute_reward(traj.text, gt, cfg)
            traj.reward = reward
            traj.is_correct = is_correct
            traj.final_answer = pred
            stats["total_rollouts"] += 1
            if is_correct:
                stats["total_correct"] += 1

        rewards = [t.reward for t in trajectories]
        advantages = compute_grpo_advantages(rewards)

        # Policy gradient update with gradient accumulation
        # IMPORTANT: backward() is called per-trajectory to free activation memory.
        # Accumulating the computation graph across G=8 trajectories would OOM.
        model.train()

        # Count trajectories that will contribute to the update (non-zero advantage)
        valid = [(t, a) for t, a in zip(trajectories, advantages) if abs(a) >= 1e-8]
        loss_count = len(valid)

        if loss_count > 0:
            optimizer.zero_grad()
            loss_sum = 0.0
            kl_sum = 0.0

            # Microbatch the policy update: each microbatch does one batched forward
            # (more efficient than one-at-a-time) and one backward (frees activations
            # before moving on to the next microbatch).
            for mb_start in range(0, loss_count, microbatch_size):
                mb = valid[mb_start : mb_start + microbatch_size]
                mb_trajs = [t for t, _ in mb]
                mb_advs = [a for _, a in mb]
                prompt_ids_list = [t.prompt_ids for t in mb_trajs]
                completion_ids_list = [t.completion_ids for t in mb_trajs]

                # Reference logprobs (LoRA disabled, no grad)
                log_p_ref_list = compute_ref_logprobs_batch(
                    model, prompt_ids_list, completion_ids_list, pad_token_id
                )

                # Active policy logprobs (graph kept for backward)
                log_p_list = compute_logprobs_batch(
                    model, prompt_ids_list, completion_ids_list, pad_token_id
                )

                mb_loss = torch.tensor(0.0, device=model.device)
                mb_kl = 0.0
                for log_p, log_p_ref, adv in zip(log_p_list, log_p_ref_list, mb_advs):
                    pg = -(adv * log_p.sum()) / loss_count
                    kl = compute_kl_k3(log_p, log_p_ref).sum() / loss_count
                    mb_loss = mb_loss + pg + cfg.kl_coef * kl
                    mb_kl += kl.item()

                mb_loss.backward()
                loss_sum += mb_loss.item()
                kl_sum += mb_kl
                del log_p_list, log_p_ref_list, mb_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            stats["step_losses"].append(loss_sum)
            stats.setdefault("step_kl", []).append(kl_sum)

        # Running stats
        stats["running_reward"].append(np.mean(rewards))
        stats["running_acc"].append(sum(t.is_correct for t in trajectories) / G)

        recent_acc = np.mean(stats["running_acc"][-50:]) if stats["running_acc"] else 0
        recent_reward = np.mean(stats["running_reward"][-50:]) if stats["running_reward"] else 0
        progress.set_postfix({
            "acc": f"{recent_acc:.1%}",
            "reward": f"{recent_reward:.2f}",
            "total_acc": f"{stats['total_correct']}/{stats['total_rollouts']}",
        })

        if (q_idx + 1) % cfg.log_every == 0:
            recent_kl = np.mean(stats.get("step_kl", [])[-50:]) if stats.get("step_kl") else 0
            log(f"[Q {q_idx+1}/{len(questions)}] recent_acc={recent_acc:.1%} "
                f"recent_reward={recent_reward:.3f} kl={recent_kl:.4f} "
                f"total_correct={stats['total_correct']}/{stats['total_rollouts']}")

        if (q_idx + 1) % cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_q{q_idx+1}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "q_idx": q_idx,
                "stats": stats,
            }, ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, "final_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "stats": stats,
    }, final_path)
    log(f"Saved final model: {final_path}")

    final_acc = stats["total_correct"] / stats["total_rollouts"] if stats["total_rollouts"] > 0 else 0
    log(f"\nFinal training accuracy: {final_acc:.1%} ({stats['total_correct']}/{stats['total_rollouts']})")

    log_file.close()
    return stats
