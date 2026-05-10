from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 32
    rollouts_per_q: int = 4
    train_questions: int = 400
    eval_questions: int = 500
    dataset_variant: str = "full"

    router_max_tokens: int = 300
    solver_max_tokens: int = 200
    synthesis_max_tokens: int = 64
    max_subgoals: int = 8
    router_temperature: float = 1.0
    solver_temperature: float = 1.0

    router_weight: float = 0.3
    solver_weight: float = 0.5
    outcome_weight: float = 0.2
    router_weight_decay: float = 0.95

    epochs: int = 1
    learning_rate: float = 1e-5
    checkpoint_every: int = 50
    log_every: int = 10

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    use_judge: bool = True
    judge_model: str = "gpt-4o-mini"
    batch_judge_size: int = 10

    device: str = "cuda"
    output_dir: str = "."
    save_rollout_traces: bool = True
    rollout_trace_path: str = ""
    use_answer_synthesis: bool = False
    router_prompt_hardening: bool = False
    plan_parse_repair: bool = False
    outcome_credit_all_steps: bool = False

    @property
    def total_per_batch(self) -> int:
        return self.batch_size * self.rollouts_per_q


def get_config() -> Config:
    return Config()
