#!/usr/bin/env bash
# Create + activate a virtualenv and install dependencies.
set -euo pipefail

python -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
