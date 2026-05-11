# Judge Ops

This directory tracks the remote judge deployment for `router_solver_v2` end to end.

## Layout

- `env/`
  - local copies of deployment and training environment files
- `manifests/`
  - deployment manifest, provenance, and progress tracking
- `nginx/`
  - reverse-proxy template for the public endpoint
- `scripts/`
  - GCP VM creation, bundle push, remote bootstrap, smoke test, and training launch scripts
- `systemd/`
  - `vllm` service template

## Intended flow

1. Copy the example env files in `env/` to `*.local` files and fill them in.
2. Run `scripts/gcp_create_judge_vm.sh` from your local machine.
3. Point DNS at the static IP that script prints.
4. Push this bundle to the VM with `scripts/push_judge_bundle.sh`.
5. SSH into the VM and run `sudo bash ~/judge_ops/scripts/bootstrap_judge_vm.sh`.
6. Run `scripts/smoke_test_remote_judge.sh` from the training machine.
7. Launch the training job with `scripts/run_slim_g6.sh`.

## Notes

- The remote service is designed around an OpenAI-compatible `/v1/chat/completions` endpoint.
- The repo client still uses `OLLAMA_*` env var names for compatibility, but the endpoint can be `vLLM`.
- `router_solver_v2` slim mode now mirrors the original `router_solver` selection rule:
  - take the first `1/8` of GSM8K train
  - then apply the existing answer parsing filter
- Provenance for that slim dataset is documented in [router_solver/slim_dataset_provenance.md](/home/pvd2112/rl_final_project/router_solver/slim_dataset_provenance.md).
- In projects that disallow external VM IPs, the intended public path is:
  - no external IP on the GPU VM
  - Cloud NAT for egress
  - external HTTP(S) load balancer in front of NGINX
