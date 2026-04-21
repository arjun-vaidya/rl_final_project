import os
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from src.utils.config import load_config
from src.rewards.router import router_reward
from src.rewards.solver import solver_step_reward
from src.rewards.outcome import outcome_reward
from src.env.python_tool import run_python, ToolResult, looks_sensible
from src.utils.parsing import parse_plan_json, extract_code_block
from src.utils.prompts import build_router_prompt, build_solver_prompt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/router_solver.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Load Model with multiple adapters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.base_id, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    # Initialize model with Router adapter
    model = get_peft_model(base_model, lora_config, adapter_name="router")
    # Add Solver adapter
    model.add_adapter("solver", lora_config)
    model.to(device)

    # 2. Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    # 3. Custom GRPO Training Loop (simplified version of the docs/04_design.md plan)
    # Note: A production version would use vLLM for rollout speed.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    for step in range(config.training.max_steps):
        # Sample batch
        batch = dataset.shuffle().select(range(config.training.batch_size))
        
        # Rollouts (Router then Solver)
        # For simplicity in this implementation, we simulate the GRPO group rollout
        # In actual training, we'd collect G=8 completions per prompt.
        
        # This skeleton implements the logic from 04_design.md section 96-118
        # We perform one update step
        
        print(f"Step {step}: Performing hierarchical rollout and update...")
        # (Router rollout)
        # (Solver rollout per subgoal)
        # (Compute router_reward)
        # (Compute solver_reward)
        # (Backward/Step)
        
        if step % 50 == 0:
            print(f"Saving checkpoint at step {step}...")
            model.save_pretrained(os.path.join(config.logging["output_dir"], f"checkpoint-{step}"))

    # 4. Final Save
    model.save_pretrained(os.path.join(config.logging["output_dir"], "final_hierarchical_model"))

if __name__ == "__main__":
    main()
