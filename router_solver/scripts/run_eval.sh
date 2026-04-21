#!/usr/bin/env bash
# Evaluate a checkpoint on val or test.
# Usage: scripts/run_eval.sh <checkpoint_path> <split: val|test>
set -euo pipefail

CKPT="${1:?checkpoint path required}"
SPLIT="${2:-val}"

python -m src.eval.evaluate --checkpoint "$CKPT" --split "$SPLIT"
