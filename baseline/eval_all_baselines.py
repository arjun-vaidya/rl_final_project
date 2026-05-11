#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, showing your reasoning "
    "clearly. End with the final answer in \\boxed{} format."
)

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_number(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def extract_answer(text):
    """Prefer the last \\boxed{N}; otherwise take the last number in the text."""
    if not text:
        return None
    boxed = BOXED_RE.findall(text)
    if boxed:
        n = parse_number(boxed[-1])
        if n is not None:
            return n
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def build_prompt(question, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def write_qa_entry(f, idx, question, ground_truth, predicted, completion, is_correct):
    f.write(f"=== q{idx} (correct={is_correct}) ===\n")
    f.write(f"Ground truth: {ground_truth}\n")
    f.write(f"Predicted:    {predicted}\n")
    f.write(f"Question:\n{question}\n\n")
    f.write(f"Model output:\n{completion}\n")
    f.write("-" * 80 + "\n\n")


def evaluate_model(name, model, tokenizer, questions, ground_truths,
                   max_tokens=512, batch_size=32, log_every=10, qa_log_path=None):
    # Left-pad so generate() works correctly for decoder-only LMs in batched mode.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    results = {"model_name": name, "correct": 0, "total": 0, "details": []}

    qa_log = None
    if qa_log_path and log_every > 0:
        os.makedirs(os.path.dirname(qa_log_path) or ".", exist_ok=True)
        qa_log = open(qa_log_path, "w")
        qa_log.write(f"# Q/A log for {name}\n")
        qa_log.write(f"# Logged every {log_every} questions out of {len(questions)}\n")
        qa_log.write("=" * 80 + "\n\n")

    progress = tqdm(range(0, len(questions), batch_size), desc=f"Evaluating {name}")
    for start in progress:
        end = min(start + batch_size, len(questions))
        q_batch = questions[start:end]
        gt_batch = ground_truths[start:end]

        prompts = [build_prompt(q, tokenizer) for q in q_batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        completions = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for i, (question, ground_truth, completion) in enumerate(zip(q_batch, gt_batch, completions)):
            predicted = extract_answer(completion)
            gt = parse_number(ground_truth)
            is_correct = predicted is not None and gt is not None and abs(predicted - gt) < 1e-6

            results["total"] += 1
            results["correct"] += int(is_correct)
            results["details"].append({
                "question": question[:100],
                "ground_truth": ground_truth,
                "predicted": completion[:150],
                "correct": is_correct,
            })

            idx = start + i
            if qa_log is not None and idx % log_every == 0:
                write_qa_entry(qa_log, idx, question, ground_truth, predicted, completion, is_correct)
                qa_log.flush()

        progress.set_postfix(acc=f"{results['correct']}/{results['total']} "
                                 f"({results['correct'] / results['total']:.1%})")

    results["accuracy"] = results["correct"] / results["total"]
    if qa_log is not None:
        qa_log.write(f"\nFinal accuracy: {results['correct']}/{results['total']} "
                     f"({results['accuracy']:.1%})\n")
        qa_log.close()
    return results


def load_gsm8k_test(num_questions=None):
    dataset = load_dataset("gsm8k", "main")["test"]
    n = len(dataset) if num_questions is None else min(num_questions, len(dataset))

    questions = [dataset[i]["question"] for i in range(n)]
    ground_truths = []
    for i in range(n):
        last_line = dataset[i]["answer"].split("\n")[-1].strip()
        match = re.search(r"####\s*([-\d.]+)", last_line)
        ground_truths.append(match.group(1) if match else dataset[i]["answer"])
    return questions, ground_truths


def load_hf_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_rl_model(base_model_id, checkpoint_path):
    """Wrap the base model in the same LoRA adapter used at training, then load
    the checkpoint. Must match linear_reasoning/src/config.py -- mismatches would
    silently leave the base model untouched under strict=False."""
    from peft import LoraConfig, get_peft_model

    model, tokenizer = load_hf_model(base_model_id)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    ))

    logging.info(f"Loading RL checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    _, unexpected = model.load_state_dict(state, strict=False)

    lora_keys = [k for k in state if "lora_" in k]
    if not lora_keys:
        raise RuntimeError(f"{checkpoint_path} has no LoRA weights -- was training run with use_lora=False?")
    dropped = [k for k in unexpected if "lora_" in k]
    if dropped:
        raise RuntimeError(f"{len(dropped)} LoRA keys failed to map (e.g. {dropped[:3]}); "
                           "LoRA config here doesn't match training.")
    logging.info(f"  loaded {len(lora_keys)} LoRA tensors")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=None, help="Number of test questions (default: all)")
    parser.add_argument("--output_dir", default="baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_qa_every", type=int, default=10,
                        help="Write one Q/A block to the model's log file every N questions (0 disables)")
    parser.add_argument("--trained_checkpoint", default=None,
                        help="Path to an RL-trained .pt checkpoint (e.g. linear_reasoning/checkpoints/final_model.pt)")
    parser.add_argument("--trained_base_model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model id the trained checkpoint was fine-tuned from")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip the two Qwen baselines and only evaluate the trained checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Loading GSM8K test set...")
    questions, ground_truths = load_gsm8k_test(args.num_questions)

    models_to_run = []
    if not args.skip_baselines:
        models_to_run += [
            ("general", "Qwen2.5-1.5B-Instruct (general)", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("math_specialized", "Qwen2.5-Math-1.5B-Instruct (math-specialized)", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        ]

    all_results = {}
    for key, display_name, hf_id in models_to_run:
        logging.info(f"\n=== {display_name} ===")
        model, tokenizer = load_hf_model(hf_id)
        all_results[key] = evaluate_model(
            display_name, model, tokenizer, questions, ground_truths,
            batch_size=args.batch_size,
            log_every=args.log_qa_every,
            qa_log_path=os.path.join(args.output_dir, f"qa_log_{key}.txt"),
        )
        logging.info(f"Accuracy: {all_results[key]['accuracy']:.1%} "
                     f"({all_results[key]['correct']}/{all_results[key]['total']})")
        del model, tokenizer

    if args.trained_checkpoint:
        ckpt_stem = os.path.splitext(os.path.basename(args.trained_checkpoint))[0]
        display_name = f"{args.trained_base_model} + RLVR ({ckpt_stem})"
        logging.info(f"\n=== {display_name} ===")
        model, tokenizer = load_rl_model(args.trained_base_model, args.trained_checkpoint)
        all_results["rl_trained"] = evaluate_model(
            display_name, model, tokenizer, questions, ground_truths,
            batch_size=args.batch_size,
            log_every=args.log_qa_every,
            qa_log_path=os.path.join(args.output_dir, f"qa_log_rl_{ckpt_stem}.txt"),
        )
        logging.info(f"Accuracy: {all_results['rl_trained']['accuracy']:.1%} "
                     f"({all_results['rl_trained']['correct']}/{all_results['rl_trained']['total']})")
        del model, tokenizer

    output_file = os.path.join(args.output_dir, "baseline_results_all.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nAll results saved to {output_file}")

    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for res in all_results.values():
        logging.info(f"{res['model_name']}: {res['accuracy']:.1%} ({res['correct']}/{res['total']})")


if __name__ == "__main__":
    main()
