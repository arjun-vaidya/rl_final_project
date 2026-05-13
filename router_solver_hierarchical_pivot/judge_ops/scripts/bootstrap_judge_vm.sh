#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo on the remote judge VM." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/env/judge.env"
if [[ -f "${ROOT_DIR}/env/judge.env.local" ]]; then
  ENV_FILE="${ROOT_DIR}/env/judge.env.local"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy env/judge.env.example to env/judge.env first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${JUDGE_DOMAIN:?}"
: "${LETSENCRYPT_EMAIL:?}"
: "${JUDGE_MODEL:?}"
: "${JUDGE_API_KEY:?}"
: "${JUDGE_PORT:?}"
: "${JUDGE_DTYPE:?}"
: "${JUDGE_GPU_MEMORY_UTILIZATION:?}"
: "${JUDGE_MAX_MODEL_LEN:?}"
: "${JUDGE_EXTRA_ARGS:=}"
: "${ENABLE_CERTBOT:=1}"

export JUDGE_DOMAIN JUDGE_PORT

apt-get update
apt-get install -y python3-venv python3-pip nginx certbot python3-certbot-nginx gettext-base

mkdir -p /opt/judge /opt/judge/logs
python3 -m venv /opt/judge/venv
/opt/judge/venv/bin/pip install --upgrade pip setuptools wheel
/opt/judge/venv/bin/pip install "vllm>=0.8.0"

cp "${ENV_FILE}" /opt/judge/judge.env
chmod 600 /opt/judge/judge.env

envsubst '${JUDGE_DOMAIN} ${JUDGE_PORT}' \
  < "${ROOT_DIR}/nginx/judge.conf.template" \
  > /etc/nginx/sites-available/judge.conf
ln -sf /etc/nginx/sites-available/judge.conf /etc/nginx/sites-enabled/judge.conf
rm -f /etc/nginx/sites-enabled/default

cp "${ROOT_DIR}/systemd/vllm-judge.service.template" /etc/systemd/system/vllm-judge.service

systemctl daemon-reload
systemctl enable vllm-judge
systemctl restart vllm-judge

nginx -t
systemctl enable nginx
systemctl restart nginx

if [[ "${ENABLE_CERTBOT}" == "1" ]]; then
  certbot --nginx --non-interactive --agree-tos --redirect -m "${LETSENCRYPT_EMAIL}" -d "${JUDGE_DOMAIN}"
fi

echo "Judge bootstrap complete."
echo "Check service: systemctl status vllm-judge --no-pager"
echo "Check nginx: systemctl status nginx --no-pager"
