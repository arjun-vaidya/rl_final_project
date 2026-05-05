#!/usr/bin/env python3
"""
Evaluate the Router-Solver hierarchical model on GSM8K.
Loads base model + LoRA adapters and runs inference.
"""
import argparse
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.env.gsm8k_loader import load_gsm8k_test
from src.agents.router_solver_agent import RouterSolverAgent
from src.rewards.outcome import outcome_reward
from src.utils.config import load_config, GlobalConfig, ModelConfig, TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint with router/solver adapters")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--num_test", type=int, default=25, help="Number of problems to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    # Load base model
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)

    # Load LoRA adapters
    print(f"Loading LoRA adapters from {args.checkpoint}...")
    router_adapter_path = f"{args.checkpoint}/router_lora"
    solver_adapter_path = f"{args.checkpoint}/solver_lora"

    model = PeftModel.from_pretrained(
        model,
        router_adapter_path,
        adapter_name="router_lora"
    )

    # Try loading solver adapter if it exists
    try:
        model.load_adapter(solver_adapter_path, adapter_name="solver_lora")
    except Exception as e:
        print(f"Solver adapter not found: {e}")

    # Load config
    try:
        config = load_config("configs/router_solver.yaml")
    except:
        print("Warning: Could not load config file, using defaults")
        config = GlobalConfig(
            model=ModelConfig(base_id=base_model_id),
            training=TrainingConfig()
        )

    # Increase token budgets for evaluation
    config.rollout.router_max_tokens = 512
    config.rollout.solver_max_tokens = 512
    config.rollout.max_subgoals = 10

    print(f"Using config: router_tokens={config.rollout.router_max_tokens}, solver_tokens={config.rollout.solver_max_tokens}, max_subgoals={config.rollout.max_subgoals}")

    # Create agent
    agent = RouterSolverAgent(model, tokenizer, config, device=device)

    # Load test data
    problems = load_gsm8k_test()[:args.num_test]
    print(f"Evaluating on {len(problems)} problems...")

    correct = 0
    results = []

    for item in tqdm(problems, desc="Evaluating"):
        question = item.question
        gt = item.numeric_answer

        try:
            # Generate rollout with greedy decoding
            rollouts = agent.batch_rollouts([question], num_rollouts=1, do_sample=False)
            if rollouts:
                rollout = rollouts[0]
                # Extract final answer from trajectory
                trajectory = "\n".join([step.output for step in rollout.steps])

                # Check if answer is correct
                is_correct = outcome_reward(trajectory, gt)
                if is_correct:
                    correct += 1

                results.append({
                    "question": question,
                    "gt": gt,
                    "correct": is_correct,
                    "trajectory": trajectory
                })
            else:
                results.append({
                    "question": question,
                    "gt": gt,
                    "correct": False,
                    "trajectory": "Failed to generate"
                })
        except Exception as e:
            results.append({
                "question": question,
                "gt": gt,
                "correct": False,
                "trajectory": f"Error: {str(e)}"
            })

    accuracy = correct / len(problems) if problems else 0.0

    print(f"\n{'='*50}")
    print(f"Results for {args.split} set ({len(problems)} problems):")
    print(f"Correct: {correct}/{len(problems)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")

    # Save results
    output_path = f"eval_router_solver_{args.split}_{len(problems)}.json"
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "results": results
        }, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
