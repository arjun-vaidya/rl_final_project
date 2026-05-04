import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from src.utils.config import load_config
from src.rewards.outcome import outcome_reward
from src.env.gsm8k_loader import extract_numeric_answer

# Load environment variables from .env file
load_dotenv()


class RewardWeightedTrainer(Trainer):
    """Custom Trainer that weights loss by rewards."""

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract rewards if present
        rewards = inputs.pop("rewards", None)

        # Standard language modeling loss
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Cross entropy loss per token
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        seq_loss = token_loss.view(shift_labels.size(0), -1).mean(dim=1)

        # Weight by rewards (higher reward = higher weight to reinforce correct behavior)
        if rewards is not None:
            rewards = rewards.float().to(seq_loss.device)
            # Scale: reward=1.0 means full weight, reward=0.0 means zero weight
            loss = (seq_loss * rewards).mean()
        else:
            loss = seq_loss.mean()

        return (loss, outputs) if return_outputs else loss


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

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    # 3. Load and preprocess dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def preprocess_example(example):
        question = example["question"]
        answer = example["answer"]
        numeric_answer = extract_numeric_answer(answer)

        prompt = f"Question: {question}\nReasoning:"
        # Split answer into execution and final answer
        answer_text = answer.split("####")[-1].strip() if "####" in answer else answer

        return {
            "prompt": prompt,
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
        }

    dataset = dataset.map(preprocess_example, remove_columns=dataset.column_names)

    # 4. Create tokenized dataset for training
    def tokenize_and_reward(examples):
        prompts = examples["prompt"]
        answers = examples["answer_text"]
        numeric_answers = examples["numeric_answer"]

        # Simple approach: concatenate prompt with answer, tokenize, compute reward
        full_texts = [p + a for p, a in zip(prompts, answers)]

        # Tokenize with smaller max_length to save memory
        tokenized = tokenizer(
            full_texts,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # Return lists, not tensors yet
        )

        # Compute rewards based on whether answer extraction would work
        rewards = []
        for answer_text, numeric_answer in zip(answers, numeric_answers):
            # Simple heuristic: if the answer text contains the numeric answer, give reward
            if numeric_answer is not None and str(numeric_answer) in answer_text:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        tokenized["rewards"] = rewards
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply tokenization
    print("Tokenizing dataset...")
    train_dataset = dataset.map(
        tokenize_and_reward,
        batched=True,
        batch_size=32,
        remove_columns=["prompt", "answer_text", "numeric_answer"]
    )

    # Convert to torch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "rewards"])

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.logging["output_dir"],
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.training.learning_rate,
        num_train_epochs=1,
        max_steps=config.training.max_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
    )

    # 6. Initialize Trainer
    trainer = RewardWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(config.logging["output_dir"], "final_model"))
    print("Training complete!")


if __name__ == "__main__":
    main()
