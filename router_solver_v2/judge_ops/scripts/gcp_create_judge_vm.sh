#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/env/gcp.env"
if [[ -f "${ROOT_DIR}/env/gcp.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/env/gcp.env.local"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy env/gcp.env.example first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${GCP_PROJECT:?}"
: "${GCP_ZONE:?}"
: "${GCP_VM_NAME:?}"
: "${GCP_MACHINE_TYPE:?}"
: "${GCP_ACCELERATOR:?}"
: "${GCP_ACCELERATOR_COUNT:?}"
: "${GCP_BOOT_DISK_GB:?}"
: "${GCP_IMAGE_FAMILY:?}"
: "${GCP_IMAGE_PROJECT:?}"
: "${GCP_NETWORK_TAG:?}"
: "${GCP_STATIC_IP_NAME:?}"
: "${GCP_ALLOWED_CIDR:?}"

REGION="${GCP_REGION:-${GCP_ZONE%-*}}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

if ! gcloud compute addresses describe "${GCP_STATIC_IP_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  gcloud compute addresses create "${GCP_STATIC_IP_NAME}" --region "${REGION}"
fi

STATIC_IP="$(gcloud compute addresses describe "${GCP_STATIC_IP_NAME}" --region "${REGION}" --format='value(address)')"

if ! gcloud compute firewall-rules describe "${GCP_NETWORK_TAG}-https" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${GCP_NETWORK_TAG}-https" \
    --direction=INGRESS \
    --allow=tcp:80,tcp:443 \
    --source-ranges="${GCP_ALLOWED_CIDR}" \
    --target-tags="${GCP_NETWORK_TAG}"
fi

if ! gcloud compute firewall-rules describe "${GCP_NETWORK_TAG}-ssh" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${GCP_NETWORK_TAG}-ssh" \
    --direction=INGRESS \
    --allow=tcp:22 \
    --source-ranges="${GCP_ALLOWED_CIDR}" \
    --target-tags="${GCP_NETWORK_TAG}"
fi

if ! gcloud compute instances describe "${GCP_VM_NAME}" --zone "${GCP_ZONE}" >/dev/null 2>&1; then
  gcloud compute instances create "${GCP_VM_NAME}" \
    --zone "${GCP_ZONE}" \
    --machine-type "${GCP_MACHINE_TYPE}" \
    --accelerator "type=${GCP_ACCELERATOR},count=${GCP_ACCELERATOR_COUNT}" \
    --maintenance-policy TERMINATE \
    --restart-on-failure \
    --boot-disk-size "${GCP_BOOT_DISK_GB}GB" \
    --image-family "${GCP_IMAGE_FAMILY}" \
    --image-project "${GCP_IMAGE_PROJECT}" \
    --tags "${GCP_NETWORK_TAG}" \
    --address "${STATIC_IP}"
fi

echo "VM_NAME=${GCP_VM_NAME}"
echo "ZONE=${GCP_ZONE}"
echo "STATIC_IP=${STATIC_IP}"
echo "SSH_COMMAND=gcloud compute ssh ${GCP_VM_NAME} --zone ${GCP_ZONE}"
