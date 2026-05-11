#!/usr/bin/env python3
import argparse
import json
import torch
import logging
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_numeric_answer(text):
    if not text:
        return None
    lines = text.split('\n')
    for line in reversed(lines):
        match = re.search(r'[-+]?\d*\.?\d+', line.strip())
        if match:
            try:
                return float(match.group())
            except:
                pass
    return None

def evaluate_model(model_name, model, tokenizer, questions, ground_truths, max_tokens=200):
    """Evaluate a baseline model on questions."""
    results = {
        "model_name": model_name,
        "correct": 0,
        "total": 0,
        "details": []
    }

    device = model.device if hasattr(model, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')

    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions), desc=f"Evaluating {model_name}"):
        results["total"] += 1

        prompt = f"Solve this math problem step by step:\n\n{question}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        completion_ids = outputs[0][inputs.input_ids.shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        predicted = extract_numeric_answer(completion_text)
        gt = extract_numeric_answer(ground_truth)

        is_correct = False
        if predicted is not None and gt is not None:
            is_correct = abs(predicted - gt) < 1e-6

        if is_correct:
            results["correct"] += 1

        results["details"].append({
            "question": question[:100],
            "ground_truth": ground_truth,
            "predicted": completion_text[:150],
            "correct": is_correct
        })

    results["accuracy"] = results["correct"] / results["total"]
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=None, help="Number of test questions (default: all)")
    parser.add_argument("--output_dir", default="baseline")
    args = parser.parse_args()

    logging.info("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    if args.num_questions is None:
        num_q = len(test_data)
    else:
        num_q = min(args.num_questions, len(test_data))

    questions = [test_data[i]["question"] for i in range(num_q)]

    ground_truths = []
    for i in range(num_q):
        answer_text = test_data[i]["answer"]
        last_line = answer_text.split("\n")[-1].strip()
        match = re.search(r'####\s*([-\d.]+)', last_line)
        if match:
            ground_truths.append(match.group(1))
        else:
            ground_truths.append(answer_text)

    all_results = {}

    # Model 1: Qwen2.5-1.5B-Instruct (general baseline)
    logging.info("\n=== Model 1: Qwen2.5-1.5B-Instruct (General) ===")
    model1 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tokenizer1.pad_token_id is None:
        tokenizer1.pad_token = tokenizer1.eos_token
    results1 = evaluate_model("Qwen2.5-1.5B-Instruct (general)", model1, tokenizer1, questions, ground_truths)
    all_results["general"] = results1
    logging.info(f"Accuracy: {results1['accuracy']:.1%} ({results1['correct']}/{results1['total']})")
    del model1, tokenizer1

    # Model 2: Qwen2.5-Math-1.5B-Instruct (math-specialized)
    logging.info("\n=== Model 2: Qwen2.5-Math-1.5B-Instruct (Math-Specialized) ===")
    model2 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    if tokenizer2.pad_token_id is None:
        tokenizer2.pad_token = tokenizer2.eos_token
    results2 = evaluate_model("Qwen2.5-Math-1.5B-Instruct (math-specialized)", model2, tokenizer2, questions, ground_truths)
    all_results["math_specialized"] = results2
    logging.info(f"Accuracy: {results2['accuracy']:.1%} ({results2['correct']}/{results2['total']})")
    del model2, tokenizer2

    # Save all results
    output_file = f"{args.output_dir}/baseline_results_all.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nAll baseline results saved to {output_file}")

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("BASELINE SUMMARY")
    logging.info("="*60)
    for key, res in all_results.items():
        logging.info(f"{res['model_name']}: {res['accuracy']:.1%} ({res['correct']}/{res['total']})")

if __name__ == "__main__":
    main()
