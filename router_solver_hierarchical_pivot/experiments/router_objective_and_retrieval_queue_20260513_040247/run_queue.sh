#!/usr/bin/env bash
set -euo pipefail
cd /home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot
QUEUE_OUT="experiments/router_objective_and_retrieval_queue_20260513_040247"
LOG="$QUEUE_OUT/queue.log"
exec > >(tee -a "$LOG") 2>&1
printf "[queue] start %s\n" "$(date -Is)"
while pgrep -f "python3 scripts/debug_hard_branch_slice.py --checkpoint experiments/slim_g6_20260508_153045/phase4_final.pt" >/dev/null; do
  printf "[queue] waiting for hard eval gpu release at %s\n" "$(date -Is)"
  sleep 30
done
printf "[queue] hard eval released gpu at %s\n" "$(date -Is)"
STAGE3_DIR=$(ls -td experiments/probe_retriever_stage3_* | head -n1)
while [ ! -f "$STAGE3_DIR/probe_hard_negatives_stage3.json" ]; do
  printf "[queue] waiting for stage3 negatives at %s\n" "$(date -Is)"
  sleep 15
  STAGE3_DIR=$(ls -td experiments/probe_retriever_stage3_* | head -n1)
done
printf "[queue] stage3 negatives ready: %s\n" "$STAGE3_DIR/probe_hard_negatives_stage3.json"
python3 scripts/train_contrastive_retriever.py \
  --corpus-json experiments/probe_retrieval_corpus_20260513_011916/probe_retrieval_corpus.json \
  --negatives-json "$STAGE3_DIR/probe_hard_negatives_stage3.json" \
  --base-model intfloat/e5-small-v2 \
  --positive-mode structural \
  --device cuda \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --output-dir "$QUEUE_OUT/retriever_e5small_probe_structural_stage3" \
  --wandb-run-name "probe_retriever_structural_stage3_$(basename "$QUEUE_OUT")"
python3 scripts/diagnose_retriever_signal.py \
  --corpus-json experiments/probe_retrieval_corpus_20260513_011916/probe_retrieval_corpus.json \
  --embedding-model "$QUEUE_OUT/retriever_e5small_probe_structural_stage3" \
  --embedding-device cuda \
  --output-json "$QUEUE_OUT/diagnostics_probe_structural_stage3.json"
python3 -u main.py \
  --mode diagnose \
  --checkpoint experiments/slim_g6_20260508_153045/phase4_final.pt \
  --dataset slim \
  --execution-branch soft \
  --diagnostic-questions 10 \
  --diagnostic-rollouts-per-q 5 \
  --router-temperature 0.2 \
  --solver-temperature 0.7 \
  --router-max-tokens 300 \
  --solver-max-tokens 512 \
  --use-answer-synthesis off \
  --plan-parse-repair on \
  --strict-answer-format off \
  --router-prompt-hardening off \
  --candidate-rerank off \
  --output-dir "$QUEUE_OUT/o1_diagnose" \
  --diagnostic-output "$QUEUE_OUT/o1_diagnostics.md" \
  --wandb-run-name "router_objective_o1_$(basename "$QUEUE_OUT")"
for VARIANT in baseline all_steps; do
  if [ "$VARIANT" = baseline ]; then
    EXTRA=()
  else
    EXTRA=(--outcome-credit-all-steps on)
  fi
  python3 -u main.py \
    --mode train \
    --checkpoint experiments/slim_g6_20260508_153045/phase4_final.pt \
    --dataset slim \
    --execution-branch soft \
    --train-questions 10 \
    --rollouts-per-q 5 \
    --epochs 1 \
    --learning-rate 1e-5 \
    --router-temperature 0.2 \
    --solver-temperature 0.7 \
    --router-max-tokens 300 \
    --solver-max-tokens 512 \
    --use-answer-synthesis off \
    --plan-parse-repair on \
    --strict-answer-format off \
    --router-prompt-hardening off \
    --candidate-rerank off \
    --save-rollout-traces on \
    --output-dir "$QUEUE_OUT/o2_${VARIANT}/train" \
    --wandb-run-name "router_objective_o2_${VARIANT}_$(basename "$QUEUE_OUT")" \
    "${EXTRA[@]}"
  python3 -u main.py \
    --mode trace_rollouts \
    --checkpoint "$QUEUE_OUT/o2_${VARIANT}/train/phase4_final.pt" \
    --dataset slim \
    --execution-branch soft \
    --diagnostic-questions 10 \
    --diagnostic-rollouts-per-q 5 \
    --router-temperature 0.2 \
    --solver-temperature 0.7 \
    --router-max-tokens 300 \
    --solver-max-tokens 512 \
    --use-answer-synthesis off \
    --plan-parse-repair on \
    --strict-answer-format off \
    --router-prompt-hardening off \
    --candidate-rerank off \
    --output-dir "$QUEUE_OUT/o2_${VARIANT}/eval" \
    --wandb-run-name "router_objective_o2_${VARIANT}_eval_$(basename "$QUEUE_OUT")"
  python3 -u main.py \
    --mode taxonomy \
    --trace-input "$QUEUE_OUT/o2_${VARIANT}/eval/rollout_traces.jsonl" \
    --taxonomy-output "$QUEUE_OUT/o2_${VARIANT}/eval/taxonomy_report.md"
done
printf "[queue] complete %s\n" "$(date -Is)"
