import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    model_id: str
    dataset_name: str = "openai/gsm8k"
    learning_rate: float = 5e-6
    batch_size: int = 8
    group_size: int = 8
    max_steps: int = 500
    beta: float = 0.04
    output_dir: str = "experiments/flat_baseline"
    wandb_project: Optional[str] = "router-solver"

def load_config(path: str) -> TrainConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)

def main():
    # Placeholder for actual training loop using TRL GRPOTrainer
    # The actual implementation will need a GPU and vLLM setup.
    print("Loading config...")
    # config = load_config("configs/flat.yaml")
    print("Initializing model and tokenizer...")
    print("Starting GRPO training for Flat Baseline...")
    # trainer = GRPOTrainer(...)
    # trainer.train()

if __name__ == "__main__":
    main()
