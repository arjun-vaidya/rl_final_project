#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/env/local_judge.env"
if [[ -f "${ROOT_DIR}/env/local_judge.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/env/local_judge.env.local"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy env/local_judge.env.example first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${OLLAMA_API_URL:?}"
: "${OLLAMA_MODEL:?}"
: "${OLLAMA_API_KEY:?}"

curl -sS "${OLLAMA_API_URL}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OLLAMA_API_KEY}" \
  -d "{
    \"model\": \"${OLLAMA_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply exactly with [7]\"}],
    \"temperature\": 0.0
  }"
