# Status Update

- Date (UTC): `2026-05-08`
- Author: `peter`
- Repo root: `/home/pvd2112/rl_final_project`
- Primary repo: `/home/pvd2112/rl_final_project/router_solver_v2`

## Executive Summary

This work session covered end-to-end setup of a remote judge service for `router_solver_v2`, exposure of that judge behind a public GCP load balancer, validation of the public OpenAI-compatible endpoint, and launch of a live `slim` dataset `G=6` training run. During monitoring, the first full run exposed a concrete judge integration problem: batched step-judging prompts were exceeding the remote judge model's `4096` token context window, and the judge model also occasionally returned fewer scores than requested for multi-item batches. Those failures were reproduced, diagnosed from remote `vLLM` logs, and mitigated in code by chunking judge requests, clipping prompt fields, salvaging partial score arrays, and reducing default batch sizes.

At handoff time:

- the GCP judge infrastructure is live and reachable at `http://34.49.229.203/v1/chat/completions`
- local smoke validation of the public judge endpoint passed
- a first production run was launched, monitored, then stopped after judge batching defects were confirmed
- a patched smoke training run completed successfully with the new chunked judge path
- a second production run has been relaunched with tighter judge batch sizes and is currently active

## Training Decisions and Current Configuration

The current active training profile is:

- dataset: `slim`
- train questions: `120`
- eval questions: `100`
- `G` / rollouts per question: `6`
- epochs: `1`
- learning rate: `1e-5`
- router temperature: `0.2`
- solver temperature: `1.0`
- router max tokens: `300`
- solver max tokens: `200`
- checkpoint every: `10`
- log every: `5`
- judge: `on`

The live launch script is:

- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops/scripts/run_slim_g6.sh`

The script was patched during this session so that:

- judge environment variables are exported to the child Python process
- the run is detached via `setsid`
- a PID file is written alongside the log
- failure to detach cleanly causes the launcher to exit with an error instead of silently returning

## Code Changes in `router_solver_v2`

The following functional changes were already in place or were added during this session:

### Training and CLI plumbing

- `main.py`
  - CLI overrides for:
    - `--train-questions`
    - `--eval-questions`
    - `--rollouts-per-q`
    - `--epochs`
    - `--learning-rate`
    - `--checkpoint-every`
    - `--log-every`
    - `--router-max-tokens`
    - `--solver-max-tokens`
    - `--router-temperature`
    - `--solver-temperature`
    - `--use-judge`
    - `--output-dir`
    - `--dataset {full,slim}`
- `src/utils/config.py`
  - config fields for dataset variant, question limits, temperatures, and output dir
- `src/agents/agent.py`
  - router and solver token/temperature settings wired through from config
- `src/training/train.py`
  - checkpoint resume logic fixed
  - `use_judge=False` fallback preserved

### Slim dataset support

Real `slim` dataset mode was added to v2 and mirrors the original documented rule from `router_solver`:

- take the first `1/8` of GSM8K train
- keep only rows with a numeric `#### ...` answer

This is now exposed through `--dataset slim`.

### Judge HTTP client

- new reusable client:
  - `src/utils/openai_compat_client.py`
- `src/rewards/judge.py`
  - now uses the reusable OpenAI-compatible HTTP client
  - supports:
    - `OLLAMA_API_URL`
    - `OLLAMA_MODEL`
    - `OLLAMA_API_KEY`
    - `OLLAMA_TIMEOUT_SEC`
    - `OLLAMA_MAX_RETRIES`
    - `OLLAMA_INITIAL_DELAY_SEC`

### Judge robustness patch from this session

After the first live run, `src/rewards/judge.py` was patched to:

- chunk plan-judge requests into smaller sub-batches
- chunk step-judge requests into smaller sub-batches
- clip large prompt fields before sending them to the remote judge:
  - question text
  - step labels
  - reasoning text
- salvage partial JSON arrays by padding/truncating instead of dropping the whole batch
- default to smaller judge batch sizes:
  - `JUDGE_PLAN_BATCH_SIZE=3`
  - `JUDGE_STEP_BATCH_SIZE=3`

The local judge env was updated to pin those settings explicitly:

- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops/env/local_judge.env.local`

## GCP Judge Infrastructure

### Project and VM

- project: `medit-478122`
- zone: `us-east1-b`
- region: `us-east1`
- VM name: `router-judge-vm`
- machine type: `g2-standard-8`
- GPU: `1 x NVIDIA L4`
- image: `common-cu129-ubuntu-2204-nvidia-580` from `ml-images`
- boot disk: `200 GB pd-balanced`

### Networking reality

This project is under an org policy that blocks VM external IPs:

- `constraints/compute.vmExternalIpAccess`

As a result:

- the VM was created with `--no-address`
- a direct public NGINX endpoint on the VM was not possible
- a reserved regional IP was created first, then later deleted because it was unusable for the no-address VM path

### Outbound access

The VM initially lacked outbound internet access, which blocked package/model install. This was fixed by creating Cloud NAT:

- router: `router-judge-nat-router`
- nat: `router-judge-nat`

### Firewall

Firewall rules created:

- `router-judge-https`
- `router-judge-ssh`

### Judge service stack on the VM

Installed and started on the VM:

- `vLLM`
- `NGINX`
- systemd unit:
  - `vllm-judge.service`

Model served remotely:

- `Qwen/Qwen2.5-7B-Instruct`

Judge host settings:

- bind address: `127.0.0.1:8000`
- public ingress handled by `NGINX`
- `NGINX` configured with host-agnostic server name `_`
- TLS not yet configured because there is still no final domain/certificate

### Public ingress

Because direct VM IP was disallowed, public ingress was implemented with a global external HTTP load balancer.

LB resources created:

- unmanaged instance group: `router-judge-ig`
- health check: `router-judge-hc`
- backend service: `router-judge-backend`
- URL map: `router-judge-url-map`
- target HTTP proxy: `router-judge-http-proxy`
- forwarding rule: `router-judge-http-fr`
- global public IP: `34.49.229.203`

Backend health was verified as:

- `HEALTHY`

## Judge Bundle and Repository Tracking

A `judge_ops` bundle was created under:

- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops`

Contents include:

- deployment runbook
- env templates
- manifests
- bootstrap scripts
- GCP creation scripts
- NGINX template
- systemd template
- smoke test
- slim `G=6` run script

Tracking/manifests updated during this session:

- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops/manifests/judge_deployment_manifest.md`
- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops/manifests/judge_progress.md`
- `/home/pvd2112/rl_final_project/router_solver_v2/judge_ops/manifests/judge_provenance.md`

## Public Judge Validation

The following validations passed:

- VM-local `vLLM` call to `/v1/chat/completions`
- VM-local `NGINX` call to `/v1/models`
- public LB root path
- public LB `/v1/models`
- public LB `/v1/chat/completions`
- local repo smoke script:
  - `router_solver_v2/judge_ops/scripts/smoke_test_remote_judge.sh`

The training-side judge env is currently pointed at:

- `OLLAMA_API_URL=http://34.49.229.203/v1/chat/completions`
- `OLLAMA_MODEL=Qwen/Qwen2.5-7B-Instruct`

Auth is via bearer token in the local env file. The token value is intentionally not copied into this report.

## Training Run History

### First production launch

Initial live run:

- output dir:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_105418`

Observed behavior:

- process launched successfully
- model loaded and entered epoch 1
- after several questions, judge failures became visible in the training log

Representative failures seen in the log:

- plan judge returned fewer scores than requested, for example `5` scores for `6` items
- step judge returned fewer scores than requested
- step judge also produced `400 Client Error: Bad Request`

The first visible progress line from that run was:

- `PROGRESS | loss=-0.0019 | acc=0.0% (0/28) | valid_q=5`

This run was not left in place because its rewards were already degraded by judge failures.

### Root cause analysis

Remote `vLLM` logs showed the exact failure mode for the `400` errors:

- `VLLMValidationError`
- maximum context length: `4096`
- offending prompt contained at least `4097` input tokens

This confirmed that the batched step-judge prompt was too large for the remote model context window.

### Patched smoke run

A patched 1-question smoke training run was launched after the judge chunking fix:

- output dir:
  - `/tmp/router_solver_v2_smoke_after_judge_fix`

Observed result:

- run completed successfully
- no `400` context-overflow failures occurred in the visible output
- judge calls completed through the patched path
- one question completed with:
  - `Questions processed: 1`
  - `Total rollouts: 6`
  - `Final loss: -0.0267`
  - `Final accuracy: 0.0%`

Important note:

- with judge batch size `4`, the remote model still occasionally returned `3` scores instead of `4`
- this did not crash training because the new parser pads/truncates partial arrays
- batch-size defaults were then reduced further to `3` for stability

### Current active production relaunch

Current active run:

- PID: `155647`
- output dir:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455`
- PID file:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.pid`
- log:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.log`

State at report time:

- process alive
- local GPU occupied by the trainer
- model loaded
- dataset loaded
- epoch 1 started
- first judged question completed cleanly after the batch-size reduction to `3`
- observed post-fix first-question judge responses:
  - plans: `[10, 10, 10]` and `[10, 10, 10]`
  - steps: `[5, 7, 8]`, `[3, 7, 8]`, `[7, 8, 9]`, `[8, 9, 7]`
- no `400` context-overflow errors seen for that question
- no score-count mismatch warnings seen for that question

## Operational Notes

### Current judge limitations

- transport is still plain HTTP because no domain/TLS cert has been provisioned
- the remote judge model does not perfectly obey multi-item score-count instructions
- smaller batch sizes mitigate this but may reduce judge throughput

### Security posture

- judge auth is enforced with bearer token
- VM has no external IP
- public ingress is through the load balancer
- direct SSH is through `gcloud` / IAP path

### Recommended next monitoring targets

The next concrete checks to perform on the active run are:

- confirm the first `checkpoint_epoch0_q10.pt` is written
- record the elapsed wall-clock time to first checkpoint
- estimate whether `120` slim questions at `G=6` are operationally acceptable

## Useful Paths and Commands

### Training logs

- active log:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.log`
- tail:
  - `tail -f /home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.log`

### Training PID

- file:
  - `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.pid`
- stop:
  - `kill "$(cat /home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_112455/train.pid)"`

### VM access

- `gcloud compute ssh router-judge-vm --project=medit-478122 --zone=us-east1-b --tunnel-through-iap`

### Remote judge logs

- `sudo journalctl -u vllm-judge -n 120 --no-pager`

## Bottom Line

The judge deployment is real, public, and working. The first live training run surfaced a real context-window defect in the judge batching path, and that defect has been diagnosed and mitigated in code. A patched smoke run succeeded. A new production `slim/G=6` run is now active with tighter judge batching and should be monitored through at least the first checkpoint before treating the setup as fully stable.
