#!/usr/bin/env python3
"""
Evaluate the Flat SFT Baseline model on GSM8K.
"""
import argparse
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.env.gsm8k_loader import load_gsm8k_test
from src.rewards.outcome import outcome_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--num_test", type=int, default=25, help="Number of problems to evaluate")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for generation")
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

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    model.eval()

    # Load test data
    problems = load_gsm8k_test()[:args.num_test]
    print(f"Evaluating on {len(problems)} problems with max_tokens={args.max_tokens}...")

    correct = 0
    results = []

    for item in tqdm(problems, desc="Evaluating"):
        question = item.question
        gt = item.numeric_answer

        try:
            # Simple generation: Question -> Answer
            prompt = f"Question: {question}\nReasoning:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            full_output = prompt + completion

            # Check if answer is correct
            is_correct = outcome_reward(full_output, gt)
            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "gt": gt,
                "correct": is_correct,
                "output": full_output
            })
        except Exception as e:
            results.append({
                "question": question,
                "gt": gt,
                "correct": False,
                "output": f"Error: {str(e)}"
            })

    accuracy = correct / len(problems) if problems else 0.0

    print(f"\n{'='*50}")
    print(f"Results for {args.split} set ({len(problems)} problems):")
    print(f"Correct: {correct}/{len(problems)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")

    # Save results
    output_path = f"eval_flat_baseline_{args.split}_{len(problems)}.json"
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "max_tokens": args.max_tokens,
            "results": results
        }, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
