#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${JUDGE_SSH_TARGET:-}" ]]; then
  scp -r "${ROOT_DIR}" "${JUDGE_SSH_TARGET}:~/judge_ops"
  echo "Copied judge_ops to ${JUDGE_SSH_TARGET}:~/judge_ops"
  exit 0
fi

ENV_FILE="${ROOT_DIR}/env/gcp.env"
if [[ -f "${ROOT_DIR}/env/gcp.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/env/gcp.env.local"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Set JUDGE_SSH_TARGET or create env/gcp.env.local first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${GCP_PROJECT:?}"
: "${GCP_ZONE:?}"
: "${GCP_VM_NAME:?}"

gcloud compute scp \
  --project="${GCP_PROJECT}" \
  --zone="${GCP_ZONE}" \
  --tunnel-through-iap \
  --recurse \
  "${ROOT_DIR}" \
  "${GCP_VM_NAME}:~/judge_ops"

echo "Copied judge_ops to ${GCP_VM_NAME}:~/judge_ops via gcloud scp"
