#!/usr/bin/env python3
"""
Main script for Phase 4 training + evaluation.

Usage:
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode train_eval
    python main.py --mode diagnose
    python main.py --mode train --checkpoint checkpoint_epoch0_q50.pt
"""

import argparse
import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Load environment variables from .env file
load_dotenv()

from src.utils.config import get_config
from src.agents.agent import Agent
from src.training.train import train
from src.training.eval import evaluate, print_eval_results
from src.training.diagnostics import run_diagnostics, format_diagnostics_report, save_diagnostics_report
from src.training.taxonomy import (
    collect_rollout_traces,
    run_trace_taxonomy,
    format_taxonomy_report,
    save_taxonomy_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_model(cfg):
    """Load base model and setup LoRA."""
    print(f"\n[1/3] Loading base model: {cfg.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Done: Model loaded")

    print(f"[2/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    print(f"Done: Tokenizer loaded")

    if cfg.use_lora:
        print(f"[3/3] Setting up LoRA adapters...")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(f"  Created default adapter")

        # Create separate adapters for router and solver
        model.add_adapter("router", lora_config)
        print(f"  Created router adapter")
        model.add_adapter("solver", lora_config)
        print(f"  Created solver adapter")
        print(f"Done: LoRA adapters ready (default, router, solver)")

    return model, tokenizer


def load_data(cfg):
    """Load GSM8K dataset. Parses ground truth from '#### <answer>' format."""
    from datasets import load_dataset
    import re

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")

    def extract_numeric_answer(answer_text):
        """Extract numeric answer from GSM8K format: '#### 42' -> '42'."""
        last_line = answer_text.split("\n")[-1].strip()
        match = re.search(r'####\s*([-\d.]+)', last_line)
        if match:
            return match.group(1)
        return None

    def build_examples(split):
        questions = []
        answers = []
        for ex in split:
            gt = extract_numeric_answer(ex["answer"])
            if gt is None:
                continue
            questions.append(ex["question"])
            answers.append(gt)
        return questions, answers

    train_split = dataset["train"]
    if cfg.dataset_variant == "slim":
        train_split = train_split.select(range(len(train_split) // 8))
        print(f"Using slim train split derived from GSM8K: raw_size={len(train_split)}")

    train_qs, train_gts = build_examples(train_split)

    test_qs, test_gts = build_examples(dataset["test"])

    print(f"Train: {len(train_qs)} | Test: {len(test_qs)} | Variant: {cfg.dataset_variant}")
    print(f"Example GT: '{train_gts[0]}'")
    return train_qs, train_gts, test_qs, test_gts


def load_checkpoint_if_available(model, optimizer, checkpoint_path, train_len: int):
    resume_epoch = 0
    resume_q_idx = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                if optimizer is not None and "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                resume_epoch = int(ckpt.get("epoch", 0))
                resume_q_idx = int(ckpt.get("q_idx", -1)) + 1
                if resume_q_idx >= train_len:
                    resume_epoch += 1
                    resume_q_idx = 0
                print(f"Resumed from epoch {resume_epoch}, q {resume_q_idx}")
            else:
                model.load_state_dict(ckpt)
                print("Loaded legacy checkpoint (model only)")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    return resume_epoch, resume_q_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train_eval", choices=["train", "eval", "train_eval", "diagnose", "taxonomy", "trace_rollouts"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to load")
    parser.add_argument("--train-questions", type=int, default=None, help="Number of training questions to use")
    parser.add_argument("--eval-questions", type=int, default=None, help="Number of eval questions to use")
    parser.add_argument("--dataset", choices=["full", "slim"], default=None, help="Training dataset variant")
    parser.add_argument("--rollouts-per-q", type=int, default=None, help="GRPO group size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate")
    parser.add_argument("--router-weight", type=float, default=None, help="Initial router reward weight")
    parser.add_argument("--solver-weight", type=float, default=None, help="Solver/process reward weight")
    parser.add_argument("--outcome-weight", type=float, default=None, help="Outcome/correctness reward weight")
    parser.add_argument("--router-weight-decay", type=float, default=None, help="Per-epoch decay for the router reward weight")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint cadence in questions")
    parser.add_argument("--log-every", type=int, default=None, help="Logging cadence in questions")
    parser.add_argument("--router-max-tokens", type=int, default=None, help="Router generation cap")
    parser.add_argument("--solver-max-tokens", type=int, default=None, help="Solver generation cap")
    parser.add_argument("--synthesis-max-tokens", type=int, default=None, help="Final answer synthesis generation cap")
    parser.add_argument("--router-temperature", type=float, default=None, help="Router sampling temperature")
    parser.add_argument("--solver-temperature", type=float, default=None, help="Solver sampling temperature")
    parser.add_argument("--use-judge", choices=["on", "off"], default=None, help="Enable or disable the judge backend")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and final model")
    parser.add_argument("--save-rollout-traces", choices=["on", "off"], default=None, help="Persist rollout traces as JSONL during training")
    parser.add_argument("--rollout-trace-path", default=None, help="Optional JSONL path for rollout traces")
    parser.add_argument("--use-answer-synthesis", choices=["on", "off"], default=None, help="Use a final synthesis step over the full trace")
    parser.add_argument("--constrained-final-answer-decoding", choices=["on", "off"], default=None, help="Use a stricter numeric-only synthesis decode")
    parser.add_argument("--candidate-rerank", choices=["on", "off"], default=None, help="Rerank numeric candidates extracted from the trace")
    parser.add_argument("--trace-consistency-guard", choices=["on", "off"], default=None, help="Reject synthesis answers not supported by the trace")
    parser.add_argument("--answer-bearing-step-hint", choices=["on", "off"], default=None, help="Hint the synthesis prompt with the most answer-like subgoal")
    parser.add_argument("--heuristic-final-selector", choices=["on", "off"], default=None, help="Select the final answer with a deterministic heuristic over trace candidates")
    parser.add_argument("--heuristic-final-selector-refined", choices=["on", "off"], default=None, help="Use a more conservative heuristic selector that only overrides synthesis on higher-confidence candidates")
    parser.add_argument("--guarded-heuristic-fallback", choices=["on", "off"], default=None, help="Fallback to the heuristic selector only when synthesis is unsupported by the trace")
    parser.add_argument("--synthesis-self-consistency", choices=["on", "off"], default=None, help="Sample multiple synthesis answers and majority-vote the numeric result")
    parser.add_argument("--synthesis-self-consistency-samples", type=int, default=None, help="Number of synthesis samples for self-consistency voting")
    parser.add_argument("--router-prompt-hardening", choices=["on", "off"], default=None, help="Use a stricter router prompt without enabling repair fallback")
    parser.add_argument("--plan-parse-repair", choices=["on", "off"], default=None, help="Enable parser repair fallback for router plans")
    parser.add_argument("--strict-answer-format", choices=["on", "off"], default=None, help="Require strict final-answer extraction pattern during decoding")
    parser.add_argument("--outcome-credit-all-steps", choices=["on", "off"], default=None, help="Distribute outcome credit across all solver steps during training")
    parser.add_argument("--diagnostic-questions", type=int, default=10, help="Questions per split for diagnostics")
    parser.add_argument("--diagnostic-rollouts-per-q", type=int, default=6, help="Rollouts per question for stochastic diagnostics")
    parser.add_argument("--diagnostic-output", default=None, help="Optional markdown path for diagnostics report")
    parser.add_argument("--trace-input", default=None, help="Path to rollout_traces.jsonl for offline taxonomy analysis")
    parser.add_argument("--taxonomy-output", default=None, help="Optional markdown path for taxonomy report")
    parser.add_argument("--taxonomy-max-failures", type=int, default=50, help="Maximum failed rollouts to include in the taxonomy result")
    args = parser.parse_args()

    cfg = get_config()
    if args.train_questions is not None:
        cfg.train_questions = args.train_questions
    if args.eval_questions is not None:
        cfg.eval_questions = args.eval_questions
    if args.dataset is not None:
        cfg.dataset_variant = args.dataset
    if args.rollouts_per_q is not None:
        cfg.rollouts_per_q = args.rollouts_per_q
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.router_weight is not None:
        cfg.router_weight = args.router_weight
    if args.solver_weight is not None:
        cfg.solver_weight = args.solver_weight
    if args.outcome_weight is not None:
        cfg.outcome_weight = args.outcome_weight
    if args.router_weight_decay is not None:
        cfg.router_weight_decay = args.router_weight_decay
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.router_max_tokens is not None:
        cfg.router_max_tokens = args.router_max_tokens
    if args.solver_max_tokens is not None:
        cfg.solver_max_tokens = args.solver_max_tokens
    if args.synthesis_max_tokens is not None:
        cfg.synthesis_max_tokens = args.synthesis_max_tokens
    if args.router_temperature is not None:
        cfg.router_temperature = args.router_temperature
    if args.solver_temperature is not None:
        cfg.solver_temperature = args.solver_temperature
    if args.use_judge is not None:
        cfg.use_judge = args.use_judge == "on"
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.save_rollout_traces is not None:
        cfg.save_rollout_traces = args.save_rollout_traces == "on"
    if args.rollout_trace_path is not None:
        cfg.rollout_trace_path = args.rollout_trace_path
    if args.use_answer_synthesis is not None:
        cfg.use_answer_synthesis = args.use_answer_synthesis == "on"
    if args.constrained_final_answer_decoding is not None:
        cfg.constrained_final_answer_decoding = args.constrained_final_answer_decoding == "on"
    if args.candidate_rerank is not None:
        cfg.candidate_rerank = args.candidate_rerank == "on"
    if args.trace_consistency_guard is not None:
        cfg.trace_consistency_guard = args.trace_consistency_guard == "on"
    if args.answer_bearing_step_hint is not None:
        cfg.answer_bearing_step_hint = args.answer_bearing_step_hint == "on"
    if args.heuristic_final_selector is not None:
        cfg.heuristic_final_selector = args.heuristic_final_selector == "on"
    if args.heuristic_final_selector_refined is not None:
        cfg.heuristic_final_selector_refined = args.heuristic_final_selector_refined == "on"
    if args.guarded_heuristic_fallback is not None:
        cfg.guarded_heuristic_fallback = args.guarded_heuristic_fallback == "on"
    if args.synthesis_self_consistency is not None:
        cfg.synthesis_self_consistency = args.synthesis_self_consistency == "on"
    if args.synthesis_self_consistency_samples is not None:
        cfg.synthesis_self_consistency_samples = args.synthesis_self_consistency_samples
    if args.router_prompt_hardening is not None:
        cfg.router_prompt_hardening = args.router_prompt_hardening == "on"
    if args.plan_parse_repair is not None:
        cfg.plan_parse_repair = args.plan_parse_repair == "on"
    if args.strict_answer_format is not None:
        cfg.strict_answer_format = args.strict_answer_format == "on"
    if args.outcome_credit_all_steps is not None:
        cfg.outcome_credit_all_steps = args.outcome_credit_all_steps == "on"
    if cfg.save_rollout_traces and not cfg.rollout_trace_path:
        cfg.rollout_trace_path = os.path.join(cfg.output_dir, "rollout_traces.jsonl")

    if args.mode == "taxonomy":
        trace_input = args.trace_input or cfg.rollout_trace_path
        if not trace_input:
            raise ValueError("taxonomy mode requires --trace-input or a configured rollout trace path")
        print("\nRunning offline rollout taxonomy...")
        taxonomy = run_trace_taxonomy(trace_input, max_failures=args.taxonomy_max_failures)
        print(format_taxonomy_report(taxonomy))

        taxonomy_output = args.taxonomy_output
        if taxonomy_output is None:
            taxonomy_output = os.path.join(os.path.dirname(trace_input) or ".", "taxonomy_report.md")
        save_taxonomy_report(taxonomy, taxonomy_output)
        print(f"Taxonomy report saved: {taxonomy_output}")
        print("\nDone!")
        return

    print("\n" + "="*70)
    print("Config for the Script")
    print("="*70)
    print(f"Batch: {cfg.batch_size} | Rollouts/Q: {cfg.rollouts_per_q} | Total/batch: {cfg.total_per_batch}")
    print(f"Dataset: {cfg.dataset_variant} | Train questions: {cfg.train_questions} | Eval questions: {cfg.eval_questions}")
    print(f"Epochs: {cfg.epochs} | LR: {cfg.learning_rate} | Use judge: {cfg.use_judge}")
    print(
        f"Reward weights: router={cfg.router_weight} "
        f"solver={cfg.solver_weight} outcome={cfg.outcome_weight} "
        f"| router_decay={cfg.router_weight_decay}"
    )
    print(f"Router max tokens: {cfg.router_max_tokens} @ temp {cfg.router_temperature}")
    print(f"Solver max tokens: {cfg.solver_max_tokens} @ temp {cfg.solver_temperature}")
    print(
        f"Synthesis max tokens: {cfg.synthesis_max_tokens} | Use synthesis: {cfg.use_answer_synthesis} | "
        f"Constrained final decode: {cfg.constrained_final_answer_decoding} | Candidate rerank: {cfg.candidate_rerank} | "
        f"Strict answer format: {cfg.strict_answer_format}"
    )
    print(
        "Trace consistency guard: "
        f"{cfg.trace_consistency_guard} | Answer-bearing step hint: {cfg.answer_bearing_step_hint} | "
        f"Heuristic selector: {cfg.heuristic_final_selector} | Heuristic selector refined: {cfg.heuristic_final_selector_refined} | "
        f"Guarded heuristic fallback: {cfg.guarded_heuristic_fallback} | "
        f"Synthesis self-consistency: {cfg.synthesis_self_consistency} ({cfg.synthesis_self_consistency_samples})"
    )
    print(f"Checkpoint every: {cfg.checkpoint_every} | Output dir: {cfg.output_dir}")
    print(f"Rollout traces: {cfg.save_rollout_traces} | Trace path: {cfg.rollout_trace_path or '-'}")
    print(f"Router prompt hardening: {cfg.router_prompt_hardening} | Plan parse repair: {cfg.plan_parse_repair} | Outcome credit all steps: {cfg.outcome_credit_all_steps}")
    print()

    # Load model
    model, tokenizer = load_model(cfg)

    agent = Agent(
        model,
        tokenizer,
        router_max_tokens=cfg.router_max_tokens,
        solver_max_tokens=cfg.solver_max_tokens,
        synthesis_max_tokens=cfg.synthesis_max_tokens,
        router_temperature=cfg.router_temperature,
        solver_temperature=cfg.solver_temperature,
        use_answer_synthesis=cfg.use_answer_synthesis,
        constrained_final_answer_decoding=cfg.constrained_final_answer_decoding,
        candidate_rerank=cfg.candidate_rerank,
        trace_consistency_guard=cfg.trace_consistency_guard,
        answer_bearing_step_hint=cfg.answer_bearing_step_hint,
        heuristic_final_selector=cfg.heuristic_final_selector,
        heuristic_final_selector_refined=cfg.heuristic_final_selector_refined,
        guarded_heuristic_fallback=cfg.guarded_heuristic_fallback,
        synthesis_self_consistency=cfg.synthesis_self_consistency,
        synthesis_self_consistency_samples=cfg.synthesis_self_consistency_samples,
        router_prompt_hardening=cfg.router_prompt_hardening,
        plan_parse_repair=cfg.plan_parse_repair,
        strict_answer_format=cfg.strict_answer_format,
    )

    # Load data
    train_qs, train_gts, test_qs, test_gts = load_data(cfg)

    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    print(f"Limited training to {len(train_qs)} questions")

    resume_epoch = 0
    resume_q_idx = 0

    # Train
    if args.mode in ["train", "train_eval"]:
        print("\nStarting training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        resume_epoch, resume_q_idx = load_checkpoint_if_available(model, optimizer, args.checkpoint, len(train_qs))

        os.makedirs(cfg.output_dir, exist_ok=True)
        train(
            agent,
            train_qs,
            train_gts,
            cfg,
            optimizer,
            start_epoch=resume_epoch,
            start_q_idx=resume_q_idx,
        )

        # Save final model
        final_ckpt = {
            "model": agent.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": cfg.epochs - 1,
            "config": cfg.__dict__,
        }
        final_ckpt_path = os.path.join(cfg.output_dir, "phase4_final.pt")
        torch.save(final_ckpt, final_ckpt_path)
        print(f"Training complete. Final checkpoint saved: {final_ckpt_path}")

    # Eval
    if args.mode in ["eval", "train_eval"]:
        if args.mode == "eval":
            load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))
        print("\nEvaluating on test set...")
        results = evaluate(agent, test_qs[:cfg.eval_questions], test_gts[:cfg.eval_questions], cfg)
        print_eval_results(results, "(Test Set)")

    if args.mode == "diagnose":
        load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))
        print("\nRunning diagnostics...")
        diagnostics = run_diagnostics(
            agent,
            train_qs,
            train_gts,
            test_qs,
            test_gts,
            cfg,
            diagnostic_questions=args.diagnostic_questions,
            diagnostic_rollouts_per_q=args.diagnostic_rollouts_per_q,
        )
        print(format_diagnostics_report(diagnostics))

        diagnostic_output = args.diagnostic_output
        if diagnostic_output is None:
            diagnostic_output = os.path.join(cfg.output_dir, "diagnostics_report.md")
        save_diagnostics_report(diagnostics, diagnostic_output)
        print(f"Diagnostics report saved: {diagnostic_output}")

    if args.mode == "trace_rollouts":
        load_checkpoint_if_available(model, None, args.checkpoint, len(train_qs))
        trace_output = args.trace_input or cfg.rollout_trace_path
        if not trace_output:
            trace_output = os.path.join(cfg.output_dir, "rollout_traces.jsonl")
        print("\nCollecting rollout traces...")
        summary = collect_rollout_traces(
            agent,
            train_qs,
            train_gts,
            args.diagnostic_questions,
            args.diagnostic_rollouts_per_q,
            trace_output,
        )
        print("Trace collection complete:")
        print(summary)
        print(f"Trace file saved: {trace_output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
