from dataclasses import dataclass


@dataclass
class Config:
    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    # Generation
    max_cot_tokens: int = 512
    temperature: float = 0.8
    eval_temperature: float = 0.0

    # Training
    rollouts_per_q: int = 8
    train_questions: int = None  # None = full GSM8K (7473)
    eval_questions: int = None   # None = full test set (1319)
    epochs: int = 1
    learning_rate: float = 1e-5

    # Reward
    correct_reward: float = 1.0
    format_reward: float = 0.5  # bonus for valid \boxed{} format
    incorrect_reward: float = 0.0

    # KL penalty against frozen reference policy (DeepSeek-R1 / GRPO style)
    kl_coef: float = 0.04

    # Training compute/memory tradeoff:
    # Number of trajectories that share one forward+backward pass during the policy update.
    # Larger = better tensor-core utilization, more memory. Set to rollouts_per_q to do the
    # whole group in one pass; set to 1 to fall back to per-trajectory (safest on small GPUs).
    train_microbatch_size: int = 4

    # Logging / Checkpointing
    log_every: int = 10
    checkpoint_every: int = 500
    output_dir: str = "linear_reasoning/experiments"

    # Device
    device: str = "cuda"


def get_config() -> Config:
    return Config()
