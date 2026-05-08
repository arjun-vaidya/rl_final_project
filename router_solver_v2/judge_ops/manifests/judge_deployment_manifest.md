# Judge Deployment Manifest

## Identity

- Deployment name: `router-judge-v1`
- Date: `2026-05-08`
- Operator: `pvd2112@columbia.edu`
- Purpose: remote OpenAI-compatible judge for `router_solver_v2`

## GCP

- Project: `medit-478122`
- Zone: `us-east1-b`
- Region: `us-east1`
- VM name: `router-judge-vm`
- Machine type: `g2-standard-8`
- GPU type/count: `nvidia-l4 x1`
- Image family/project: `common-cu129-ubuntu-2204-nvidia-580` from `ml-images`
- VM networking: `no external IP` due org policy
- Network tag: `router-judge`
- Firewall rules:
  - `router-judge-https`
  - `router-judge-ssh`
- Cloud NAT:
  - router `router-judge-nat-router`
  - nat `router-judge-nat`
- Reserved regional IP: deleted after the no-address VM + load balancer path was confirmed

## Public Endpoint

- Current plan: external HTTP(S) load balancer in front of the VM
- Current state: external HTTP load balancer active and serving traffic
- Public HTTP IP: `34.49.229.203`
- Temporary NGINX mode: HTTP on VM, host-agnostic server name `_`
- TLS mode: pending domain / certificate setup
- API auth: bearer token via `OLLAMA_API_KEY`

## Model

- Model ID: `Qwen/Qwen2.5-7B-Instruct`
- Dtype: `auto`
- GPU memory utilization: `0.90`
- Max model length: `4096`
- Extra args: `--generation-config vllm`

## Training Consumer

- Training host: `/home/pvd2112/rl_final_project`
- Repo path: `/home/pvd2112/rl_final_project/router_solver_v2`
- Env file used: `judge_ops/env/local_judge.env.local`
- Primary launch script: `judge_ops/scripts/run_slim_g6.sh`
- Active run:
  - output dir: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455`
  - pid file: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.pid`
  - launch status: process alive after judge batching fix; initial rollout generation in progress

## Validation

- VM health:
  - `nvidia-smi` OK
  - Python present
- Outbound internet:
  - initially blocked with no external IP
  - restored via Cloud NAT
- Public ingress:
  - `/v1/models` through LB returned `200`
  - `/v1/chat/completions` through LB returned `200`
  - local `smoke_test_remote_judge.sh` passed against the public IP
- Remaining validation:
  - optional HTTPS/domain setup

## Outcome

- Status: `active`
- Open issues:
  - TLS still pending a real domain
