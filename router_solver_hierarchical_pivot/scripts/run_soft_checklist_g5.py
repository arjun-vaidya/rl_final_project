#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import wandb
from dotenv import load_dotenv

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import load_model, load_data, load_checkpoint_if_available
from src.agents.agent import Agent
from src.training.taxonomy import collect_rollout_traces
from src.utils.config import get_config


load_dotenv()


def load_trace_summary(trace_path: Path):
    rows = []
    with trace_path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    total_questions = len(rows)
    majority_relaxed = sum(int(bool(row.get("group_summary", {}).get("majority_relaxed_match", False))) for row in rows)
    majority_exact = sum(int(bool(row.get("group_summary", {}).get("majority_exact_match", False))) for row in rows)
    any_relaxed = sum(int(bool(row.get("group_summary", {}).get("any_relaxed_match", False))) for row in rows)
    valid_rollouts = sum(int(row.get("group_summary", {}).get("valid_rollouts", 0)) for row in rows)
    total_rollouts = sum(int(row.get("group_summary", {}).get("total_rollouts", 0)) for row in rows)
    return {
        "records": rows,
        "question_majority_relaxed_accuracy": majority_relaxed / max(total_questions, 1),
        "question_majority_exact_accuracy": majority_exact / max(total_questions, 1),
        "question_any_relaxed_accuracy": any_relaxed / max(total_questions, 1),
        "valid_rollouts": valid_rollouts,
        "total_rollouts": total_rollouts,
        "num_questions": total_questions,
    }


def init_wandb(project: str, run_name: str):
    try:
        return wandb.init(
            project=project,
            name=run_name,
            config={
                "diagnostic_questions": 10,
                "diagnostic_rollouts_per_q": 5,
                "rollouts_per_q": 5,
                "checkpoint": "experiments/slim_g6_20260508_153045/phase4_final.pt",
            },
        )
    except Exception as e:
        print(f"W&B online init failed ({e}); falling back to offline mode.", flush=True)
        return wandb.init(
            project=project,
            name=run_name,
            mode="offline",
            config={
                "diagnostic_questions": 10,
                "diagnostic_rollouts_per_q": 5,
                "rollouts_per_q": 5,
                "checkpoint": "experiments/slim_g6_20260508_153045/phase4_final.pt",
            },
        )


def make_cfg(output_dir: Path):
    cfg = get_config()
    cfg.dataset_variant = "slim"
    cfg.train_questions = 400
    cfg.rollouts_per_q = 5
    cfg.router_max_tokens = 300
    cfg.solver_max_tokens = 512
    cfg.synthesis_max_tokens = 128
    cfg.use_answer_synthesis = False
    cfg.plan_parse_repair = True
    cfg.strict_answer_format = False
    cfg.answer_bearing_step_hint = True
    cfg.candidate_rerank = False
    cfg.router_prompt_hardening = False
    cfg.heuristic_final_selector_refined = False
    cfg.guarded_heuristic_fallback = False
    cfg.output_dir = str(output_dir)
    cfg.save_rollout_traces = True
    return cfg


def make_agent(model, tokenizer, cfg, execution_branch: str, solver_temperature: float, solver_max_tokens: int):
    return Agent(
        model,
        tokenizer,
        router_max_tokens=cfg.router_max_tokens,
        solver_max_tokens=solver_max_tokens,
        synthesis_max_tokens=cfg.synthesis_max_tokens,
        router_temperature=cfg.router_temperature,
        solver_temperature=solver_temperature,
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
        execution_branch=execution_branch,
    )


def run_variant(agent, train_qs, train_gts, variant_dir: Path, num_questions: int = 10, rollouts_per_q: int = 5):
    trace_path = variant_dir / "rollout_traces.jsonl"
    if trace_path.exists():
        summary = load_trace_summary(trace_path)
        summary["wall_time_seconds"] = None
        summary["resumed_from_cache"] = True
        return summary
    start = time.time()
    summary = collect_rollout_traces(
        agent,
        train_qs,
        train_gts,
        num_questions,
        rollouts_per_q,
        str(trace_path),
    )
    wall_time = time.time() - start
    summary["records"] = load_trace_summary(trace_path)["records"]
    summary["wall_time_seconds"] = wall_time
    summary["resumed_from_cache"] = False
    return summary


def compute_oracle(easy_summary, soft_summary):
    easy_rows = easy_summary["records"]
    soft_rows = soft_summary["records"]
    assert len(easy_rows) == len(soft_rows)
    chosen_soft = 0
    chosen_easy = 0
    oracle_relaxed = 0
    oracle_exact = 0
    for easy_row, soft_row in zip(easy_rows, soft_rows):
        easy_group = easy_row.get("group_summary", {})
        soft_group = soft_row.get("group_summary", {})
        easy_ok = bool(easy_group.get("majority_relaxed_match", False))
        soft_ok = bool(soft_group.get("majority_relaxed_match", False))
        if soft_ok and not easy_ok:
            chosen_soft += 1
            oracle_relaxed += 1
            oracle_exact += int(bool(soft_group.get("majority_exact_match", False)))
        else:
            chosen_easy += 1
            oracle_relaxed += int(easy_ok)
            oracle_exact += int(bool(easy_group.get("majority_exact_match", False)))
    n = len(easy_rows)
    easy_avg_q_time = (easy_summary.get("wall_time_seconds") or 0.0) / max(easy_summary["num_questions"], 1)
    soft_avg_q_time = (soft_summary.get("wall_time_seconds") or 0.0) / max(soft_summary["num_questions"], 1)
    oracle_time = chosen_easy * easy_avg_q_time + chosen_soft * soft_avg_q_time
    return {
        "question_majority_relaxed_accuracy": oracle_relaxed / max(n, 1),
        "question_majority_exact_accuracy": oracle_exact / max(n, 1),
        "chosen_easy_questions": chosen_easy,
        "chosen_soft_questions": chosen_soft,
        "estimated_wall_time_seconds": oracle_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--wandb-project", default="router_solver_hierarchical_pivot")
    parser.add_argument("--wandb-run-name", default="soft_checklist_g5")
    parser.add_argument("--existing-out-dir", default="/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/soft_checklist_g5_20260513_015435")
    args = parser.parse_args()

    if args.existing_out_dir:
        out_dir = Path(args.existing_out_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / args.output_root / f"soft_checklist_g5_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    init_wandb(args.wandb_project, args.wandb_run_name)

    cfg = make_cfg(out_dir)
    model, tokenizer = load_model(cfg)
    train_qs, train_gts, _test_qs, _test_gts = load_data(cfg)
    train_qs = train_qs[:cfg.train_questions]
    train_gts = train_gts[:cfg.train_questions]
    load_checkpoint_if_available(model, None, "experiments/slim_g6_20260508_153045/phase4_final.pt", len(train_qs))

    results = {}

    completed_dirs = {
        "H2": out_dir / "H2_soft_temp04",
        "H3": out_dir / "H3_soft_tokens256",
        "B1": out_dir / "B1_easy_only",
        "B2": out_dir / "B2_soft_only",
    }
    for d in completed_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    results["H2"] = run_variant(
        make_agent(model, tokenizer, cfg, execution_branch="soft", solver_temperature=0.4, solver_max_tokens=512),
        train_qs,
        train_gts,
        completed_dirs["H2"],
    )
    wandb.log({f"H2/{k}": v for k, v in results["H2"].items() if isinstance(v, (int, float))})

    results["H3"] = run_variant(
        make_agent(model, tokenizer, cfg, execution_branch="soft", solver_temperature=0.7, solver_max_tokens=256),
        train_qs,
        train_gts,
        completed_dirs["H3"],
    )
    wandb.log({f"H3/{k}": v for k, v in results["H3"].items() if isinstance(v, (int, float))})

    results["B1"] = run_variant(
        make_agent(model, tokenizer, cfg, execution_branch="easy", solver_temperature=0.7, solver_max_tokens=512),
        train_qs,
        train_gts,
        completed_dirs["B1"],
    )
    wandb.log({f"B1/{k}": v for k, v in results["B1"].items() if isinstance(v, (int, float))})

    results["B2"] = run_variant(
        make_agent(model, tokenizer, cfg, execution_branch="soft", solver_temperature=0.7, solver_max_tokens=512),
        train_qs,
        train_gts,
        completed_dirs["B2"],
    )
    wandb.log({f"B2/{k}": v for k, v in results["B2"].items() if isinstance(v, (int, float))})

    b3 = compute_oracle(results["B1"], results["B2"])
    with (out_dir / "B3_oracle_summary.json").open("w", encoding="ascii", errors="ignore") as f:
        json.dump(b3, f, indent=2, ensure_ascii=True)
    wandb.log({f"B3/{k}": v for k, v in b3.items() if isinstance(v, (int, float))})

    summary = {
        "H2": {k: v for k, v in results["H2"].items() if k != "records"},
        "H3": {k: v for k, v in results["H3"].items() if k != "records"},
        "B1": {k: v for k, v in results["B1"].items() if k != "records"},
        "B2": {k: v for k, v in results["B2"].items() if k != "records"},
        "B3": b3,
    }
    with (out_dir / "soft_checklist_summary.json").open("w", encoding="ascii", errors="ignore") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
