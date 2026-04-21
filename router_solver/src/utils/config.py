import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    base_id: str
    router_adapter_name: str = "router_lora"
    solver_adapter_name: str = "solver_lora"
    lora_r: int = 16
    lora_alpha: int = 32

@dataclass
class MemoryConfig:
    enabled: bool = False
    mode: str = "none" # none, random, similarity
    k: int = 3
    capacity: int = 1000
    min_similarity: float = 0.5
    write_gate_reward_threshold: float = 1.0

@dataclass
class TrainingConfig:
    algorithm: str = "GRPO"
    learning_rate: float = 5.0e-6
    beta: float = 0.04
    batch_size: int = 8
    group_size: int = 8
    max_steps: int = 500
    reward_mode: str = "decomposed" # outcome_only, decomposed

@dataclass
class RolloutConfig:
    max_subgoals: int = 6
    router_max_tokens: int = 256
    solver_max_tokens: int = 128

@dataclass
class GlobalConfig:
    model: ModelConfig
    training: TrainingConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: Dict[str, Any] = field(default_factory=lambda: {"output_dir": "experiments/run", "wandb_project": "router-solver"})

def load_config(path: str) -> GlobalConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
        
    # Flat/Nested conversion with safe defaults
    model = ModelConfig(**data.get("model", {}))
    training = TrainingConfig(**data.get("training", {}))
    rollout = RolloutConfig(**data.get("rollout", {}))
    memory = MemoryConfig(**data.get("memory", {}))
    logging = data.get("logging", {"output_dir": "experiments/run", "wandb_project": "router-solver"})
    
    return GlobalConfig(
        model=model,
        training=training,
        rollout=rollout,
        memory=memory,
        logging=logging
    )
