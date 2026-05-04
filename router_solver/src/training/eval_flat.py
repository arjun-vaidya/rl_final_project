import os
import torch
import wandb
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from src.utils.config import load_config
from src.rewards.outcome import outcome_reward, extract_answer_from_trajectory
from src.env.gsm8k_loader import extract_numeric_answer

load_dotenv()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="experiments/flat_baseline/final_model")
    parser.add_argument("--num_samples", type=int, default=25)
    args = parser.parse_args()

    # Initialize wandb for evaluation
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

        wandb.init(
            project="router-solver",
            name="eval-flat-baseline",
            job_type="evaluation"
        )
        
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()

    # Load a few test samples
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    print(f"\n{'='*80}")
    print(f"Testing trained flat baseline model on {args.num_samples} test examples")
    print(f"{'='*80}\n")

    correct = 0
    for i in range(min(args.num_samples, len(dataset))):
        example = dataset[i]
        question = example["question"]
        ground_truth_answer = example["answer"]
        gt_numeric = extract_numeric_answer(ground_truth_answer)

        # Format prompt
        prompt = f"Question: {question}\nReasoning:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Check reward
        reward = outcome_reward(completion, gt_numeric)
        correct += reward

        print(f"Example {i+1}:")
        print(f"Question: {question}")
        print(f"\nGenerated: {completion}")
        print(f"\nGround truth: {ground_truth_answer}")
        print(f"Reward (1=correct, 0=wrong): {reward}")
        print(f"-" * 80 + "\n")

    accuracy = correct / min(args.num_samples, len(dataset))
    print(f"\nAccuracy on {min(args.num_samples, len(dataset))} test samples: {accuracy:.1%}")

    if wandb_api_key:
        wandb.log({
            "eval_accuracy": accuracy,
            "num_samples": min(args.num_samples, len(dataset)),
            "correct_answers": correct,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
