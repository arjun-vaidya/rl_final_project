import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.agents.router_solver_agent import RouterSolverAgent
from src.utils.config import load_config
from peft import LoraConfig, get_peft_model

def main():
    os.environ["WANDB_MODE"] = "disabled"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use a tiny model for fast benchmarking
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    
    cfg = load_config("configs/router_solver.yaml")
    
    lora_cfg = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, lora_cfg, adapter_name=cfg.model.router_adapter_name)
    model.add_adapter(cfg.model.solver_adapter_name, lora_cfg)
    model.to(device)
    model.eval()
    # Extremely aggressive config overrides for benchmark
    cfg.rollout.router_max_tokens = 64
    cfg.rollout.solver_max_tokens = 64
    cfg.rollout.max_subgoals = 2
    
    agent = RouterSolverAgent(model, tokenizer, cfg, device=device)
    
    questions = [
        "What is 2 + 2?",
        "Write a python script to reverse a string.",
        "Calculate the 10th fibonacci number.",
        "Solve the equation 2x + 4 = 10"
    ]
    
    print("\n--- Running Batched Rollout ---")
    start = time.time()
    with torch.no_grad():
        rollouts = agent.batched_rollout(questions, do_sample=False, batch_size=4)
    batched_time = time.time() - start
    print(f"Batched Rollout Time (4 questions): {batched_time:.2f}s")
    for ro in rollouts:
        print(f"Q: {ro.question} | Steps: {len(ro.steps)} | Err: {ro.tool_error_count}")
        
    print("\n--- Running Sequential Rollout ---")
    start = time.time()
    with torch.no_grad():
        rollouts_seq = [agent.rollout(q, do_sample=False) for q in questions]
    seq_time = time.time() - start
    print(f"Sequential Rollout Time (4 questions): {seq_time:.2f}s")
    
    print(f"\nSpeedup: {seq_time / batched_time:.2f}x")

if __name__ == "__main__":
    main()
