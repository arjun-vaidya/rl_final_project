import argparse
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.env.gsm8k_loader import load_gsm8k_train, load_gsm8k_test
from src.utils.parsing import extract_code_block
from src.rewards.outcome import outcome_reward, extract_answer_from_trajectory
from src.env.python_tool import run_python, ToolResult

def evaluate_flat(model, tokenizer, problems, device="cuda"):
    """Evaluates the flat agent baseline."""
    correct = 0
    total = len(problems)
    results = []

    for item in tqdm(problems, desc="Evaluating Flat"):
        question = item.question
        gt = item.numeric_answer
        
        # Interleaved tool-use loop
        prompt = f"Question: {question}\nReasoning:"
        trajectory = prompt
        max_steps = 5
        
        for _ in range(max_steps):
            inputs = tokenizer(trajectory, return_tensors="pt").to(device)
            # Generate until </code> or <answer>
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                stop_strings=["</code>", "</answer>"],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id
            )
            
            new_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            trajectory += new_text
            
            if "</code>" in new_text:
                # Extract and run code
                code = extract_code_block(trajectory)
                if code:
                    tool_res = run_python(code)
                    trajectory += f"\nOutput: {tool_res.output}\nReasoning:"
            elif "</answer>" in new_text or tokenizer.eos_token in new_text:
                break
        
        is_correct = outcome_reward(trajectory, gt)
        if is_correct:
            correct += 1
            
        results.append({
            "question": question,
            "trajectory": trajectory,
            "gt": gt,
            "correct": is_correct
        })

    accuracy = correct / total
    return accuracy, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--mode", type=str, choices=["flat", "hierarchical"], default="flat")
    args = parser.parse_args()

    # Load data
    if args.split == "val":
        # Load a 200-problem subset as per docs/03_dataset.md
        problems = load_gsm8k_train()[:200]
    else:
        problems = load_gsm8k_test()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device)

    if args.mode == "flat":
        acc, res = evaluate_flat(model, tokenizer, problems, device)
    else:
        print("Hierarchical evaluation not yet fully implemented.")
        acc, res = 0.0, []

    print(f"\nResults for {args.split} ({args.mode}):")
    print(f"Accuracy: {acc:.2%}")

    # Save results
    output_path = f"eval_{args.split}_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump({"accuracy": acc, "results": res}, f, indent=2)
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
