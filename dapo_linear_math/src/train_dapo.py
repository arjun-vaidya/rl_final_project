import os
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from typing import List, Tuple

from agent import LinearReasoningAgent, Trajectory
from reward import compute_reward


def compute_logprobs_batch(model, prompt_ids_list, completion_ids_list, pad_token_id):
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
    logits = outputs.logits

    results = []
    for i in range(B):
        p_len = prompt_lens[i]
        c_len = completion_lens[i]
        completion_logits = logits[i, p_len - 1 : p_len - 1 + c_len]
        completion_targets = completion_ids_list[i].to(device)
        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze(1)

        eos_positions = (completion_targets == pad_token_id).nonzero(as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            first_eos = eos_positions[0].item()
            mask = torch.ones_like(token_log_probs)
            mask[first_eos + 1 :] = 0.0
            token_log_probs = token_log_probs * mask

        results.append(token_log_probs)

    return results


@torch.no_grad()
def compute_ref_logprobs_batch(model, prompt_ids_list, completion_ids_list, pad_token_id):
    with model.disable_adapter():
        return [lp.detach() for lp in compute_logprobs_batch(
            model, prompt_ids_list, completion_ids_list, pad_token_id
        )]


def compute_kl_k3(log_p, log_p_ref):
    log_ratio = log_p_ref - log_p
    return torch.exp(log_ratio) - log_ratio - 1.0


def compute_grpo_advantages(rewards):
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std = rewards_t.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    return ((rewards_t - mean) / (std + 1e-8)).tolist()


def rollout_multi_group(agent: LinearReasoningAgent, questions: List[str], gts: List[str], G: int, temperature: float):
    """Generate G rollouts for each of len(questions) questions in one batched HF call.

    Returns: list of length len(questions), each entry is a list of G Trajectories.
    """
    prompts_flat = []
    for q in questions:
        prompts_flat.extend([agent._build_prompt(q)] * G)

    results = agent._generate_batch(prompts_flat, agent.max_cot_tokens, temperature)

    grouped = []
    for qi, (q, gt) in enumerate(zip(questions, gts)):
        group = []
        for gi in range(G):
            text, prompt_ids, completion_ids = results[qi * G + gi]
            group.append(Trajectory(
                question=q,
                ground_truth=gt,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                text=text,
            ))
        grouped.append(group)
    return grouped


def train_dapo(
    agent: LinearReasoningAgent,
    questions: List[str],
    ground_truths: List[str],
    cfg,
    optimizer,
    vllm_engine=None,
):
    """DAPO-style training loop.

    Differences from vanilla GRPO:
      1. Dynamic sampling: groups where all rollouts are correct or all wrong
         are dropped, and we keep pulling new questions until the step has
         `groups_per_step` informative groups (or hits resample cap).
      2. Token-level policy gradient: loss normalizes by total tokens in the
         step, not by number of trajectories.
      3. Megabatched rollouts: B questions x G rollouts in one call.

    If `vllm_engine` is provided, rollouts are produced by vLLM and the PEFT
    adapter is synced to vLLM after every gradient step. Forward/backward for
    the policy update stay on the HF model.
    """
    model = agent.model
    G = cfg.rollouts_per_q
    pad_token_id = agent.tokenizer.pad_token_id
    microbatch_size = max(1, cfg.train_microbatch_size)
    groups_per_step = cfg.dapo_groups_per_step
    rollout_batch = cfg.dapo_rollout_batch
    max_attempts = cfg.dapo_max_resamples
    sync_every = getattr(cfg, "vllm_sync_every", 1)

    stats = {
        "total_correct": 0,
        "total_rollouts": 0,
        "kept_groups": 0,
        "dropped_all_correct": 0,
        "dropped_all_wrong": 0,
        "running_reward": [],
        "running_acc": [],
        "step_losses": [],
        "step_kl": [],
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"DAPO training: |Q|={len(questions)} G={G} groups/step={groups_per_step} rollout_batch={rollout_batch}")

    pool = list(zip(questions, ground_truths))
    pool_idx = 0
    step = 0
    progress = tqdm(total=len(pool), desc="DAPO")

    while pool_idx < len(pool):
        step += 1
        informative = []  # list of (trajs, advs)
        attempts = 0

        # Keep pulling B questions at a time until we have enough informative groups
        while len(informative) < groups_per_step and attempts < max_attempts and pool_idx < len(pool):
            batch_questions = []
            batch_gts = []
            batch_indices = []
            for _ in range(rollout_batch):
                if pool_idx >= len(pool):
                    break
                q, gt = pool[pool_idx]
                batch_questions.append(q)
                batch_gts.append(gt)
                batch_indices.append(pool_idx)
                pool_idx += 1
                attempts += 1

            if not batch_questions:
                break

            model.eval()
            if vllm_engine is not None:
                grouped_trajs = vllm_engine.generate_groups(
                    agent, batch_questions, batch_gts, G, cfg.temperature
                )
            else:
                grouped_trajs = rollout_multi_group(agent, batch_questions, batch_gts, G, cfg.temperature)

            # Score and filter each group
            for trajs in grouped_trajs:
                gt = trajs[0].ground_truth
                rewards = []
                for t in trajs:
                    r, c, p = compute_reward(t.text, gt, cfg)
                    t.reward = r
                    t.is_correct = c
                    t.final_answer = p
                    rewards.append(r)
                    stats["total_rollouts"] += 1
                    if c:
                        stats["total_correct"] += 1

                n_correct = sum(t.is_correct for t in trajs)
                if n_correct == 0:
                    stats["dropped_all_wrong"] += 1
                    continue
                if n_correct == G:
                    stats["dropped_all_correct"] += 1
                    continue

                advs = compute_grpo_advantages(rewards)
                informative.append((trajs, advs))
                stats["kept_groups"] += 1

            progress.update(len(batch_questions))

        if not informative:
            log(f"[step {step}] no informative groups after {attempts} attempts; skipping policy update")
            continue

        # Flatten for the policy update
        flat = []
        for trajs, advs in informative:
            for t, a in zip(trajs, advs):
                if abs(a) >= 1e-8:
                    flat.append((t, a))

        if not flat:
            continue

        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        kl_sum = 0.0

        # Token-level loss: count total completion tokens across the step
        total_tokens = sum(t.completion_ids.size(0) for t, _ in flat)
        scale = max(1, total_tokens)

        for mb_start in range(0, len(flat), microbatch_size):
            mb = flat[mb_start : mb_start + microbatch_size]
            mb_trajs = [t for t, _ in mb]
            mb_advs = [a for _, a in mb]
            prompt_ids_list = [t.prompt_ids for t in mb_trajs]
            completion_ids_list = [t.completion_ids for t in mb_trajs]

            log_p_ref_list = compute_ref_logprobs_batch(model, prompt_ids_list, completion_ids_list, pad_token_id)
            log_p_list = compute_logprobs_batch(model, prompt_ids_list, completion_ids_list, pad_token_id)

            mb_loss = torch.tensor(0.0, device=model.device)
            mb_kl = 0.0
            for log_p, log_p_ref, adv in zip(log_p_list, log_p_ref_list, mb_advs):
                # Token-level: contribute each token's logprob, normalized by total step tokens
                pg = -(adv * log_p.sum()) / scale
                kl = compute_kl_k3(log_p, log_p_ref).sum() / scale
                mb_loss = mb_loss + pg + cfg.kl_coef * kl
                mb_kl += kl.item()

            mb_loss.backward()
            loss_sum += mb_loss.item()
            kl_sum += mb_kl
            del log_p_list, log_p_ref_list, mb_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        stats["step_losses"].append(loss_sum)
        stats["step_kl"].append(kl_sum)

        # Push the freshly-updated LoRA to vLLM for the next rollout step.
        if vllm_engine is not None and (step % sync_every == 0):
            vllm_engine.sync_lora_from_peft(model)

        # Running stats over the trajectories we just trained on
        all_rewards = [t.reward for trajs, _ in informative for t in trajs]
        all_correct = [t.is_correct for trajs, _ in informative for t in trajs]
        if all_rewards:
            stats["running_reward"].append(float(np.mean(all_rewards)))
            stats["running_acc"].append(float(np.mean(all_correct)))

        recent_acc = float(np.mean(stats["running_acc"][-50:])) if stats["running_acc"] else 0.0
        recent_reward = float(np.mean(stats["running_reward"][-50:])) if stats["running_reward"] else 0.0
        recent_kl = float(np.mean(stats["step_kl"][-50:])) if stats["step_kl"] else 0.0
        recent_loss = float(np.mean(stats["step_losses"][-50:])) if stats["step_losses"] else 0.0
        cumulative_acc = stats["total_correct"] / stats["total_rollouts"] if stats["total_rollouts"] else 0.0

        progress.set_postfix({
            "step": step,
            "kept": stats["kept_groups"],
            "drop_c": stats["dropped_all_correct"],
            "drop_w": stats["dropped_all_wrong"],
            "acc": f"{recent_acc:.2%}",
            "reward": f"{recent_reward:.2f}",
        })

        if wandb.run is not None:
            wandb.log({
                "train/recent_accuracy": recent_acc,
                "train/recent_reward": recent_reward,
                "train/recent_kl": recent_kl,
                "train/recent_loss": recent_loss,
                "train/cumulative_accuracy": cumulative_acc,
                "train/kept_groups": stats["kept_groups"],
                "train/dropped_all_correct": stats["dropped_all_correct"],
                "train/dropped_all_wrong": stats["dropped_all_wrong"],
                "train/pool_idx": pool_idx,
                "train/step": step,
            }, step=step)

        if step % cfg.log_every == 0:
            log(f"[step {step}] kept={stats['kept_groups']} "
                f"drop_correct={stats['dropped_all_correct']} drop_wrong={stats['dropped_all_wrong']} "
                f"recent_acc={recent_acc:.2%} kl={recent_kl:.4f} loss={recent_loss:.3f}")

        if step % cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_step{step}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "pool_idx": pool_idx,
                "stats": stats,
            }, ckpt_path)
            log(f"Saved checkpoint: {ckpt_path}")

    progress.close()

    final_path = os.path.join(cfg.output_dir, "final_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "stats": stats,
    }, final_path)

    final_acc = stats["total_correct"] / stats["total_rollouts"] if stats["total_rollouts"] else 0.0
    log(f"\nFinal training rollout accuracy: {final_acc:.2%} ({stats['total_correct']}/{stats['total_rollouts']})")
    log(f"Kept groups: {stats['kept_groups']}, dropped (all correct): {stats['dropped_all_correct']}, dropped (all wrong): {stats['dropped_all_wrong']}")
    log_file.close()
    return stats
