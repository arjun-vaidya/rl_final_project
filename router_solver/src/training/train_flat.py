import os
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.utils.config import load_config
from src.rewards.outcome import outcome_reward, extract_answer_from_trajectory
from src.env.gsm8k_loader import extract_numeric_answer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/flat.yaml")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    
    # 2. Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Dataset
    def format_dataset(examples):
        # GRPO expects a "prompt" column
        prompts = [
            f"Question: {q}\nReasoning:" 
            for q in examples["question"]
        ]
        # And we pass the numeric answer for the reward function
        numeric_answers = [extract_numeric_answer(a) for a in examples["answer"]]
        return {"prompt": prompts, "numeric_answer": numeric_answers}

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_dataset, batched=True)

    # 4. Reward function for GRPOTrainer
    # GRPOTrainer reward functions take (prompts, completions, **kwargs)
    def reward_fn(prompts, completions, numeric_answer, **kwargs):
        rewards = []
        for completion, gt in zip(completions, numeric_answer):
            # Extract answer from the completion string
            rewards.append(outcome_reward(completion, gt))
        return rewards

    # 5. Trainer Configuration
    training_args = GRPOConfig(
        output_dir=config.logging["output_dir"],
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=config.training.max_steps,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        # GRPO specific
        num_generations=config.training.group_size,
        beta=config.training.beta,
        max_prompt_length=256,
        max_completion_length=768, # Total 1024 as per docs
        report_to="wandb" if config.logging.get("wandb_project") else "none",
    )

    # 6. Initialize Trainer
    trainer = GRPOTrainer(
        model=config.model.base_id, # trl can load it directly
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 7. Train
    trainer.train()
    trainer.save_model(os.path.join(config.logging["output_dir"], "final_model"))

if __name__ == "__main__":
    main()
