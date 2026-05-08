# Judge Provenance

## Source Assets

- Repo path: `/home/pvd2112/rl_final_project/router_solver_v2`
- Project: `medit-478122`
- VM: `router-judge-vm`

## Provisioning Record

- VM created with:
  - `g2-standard-8`
  - `1 x NVIDIA L4`
  - `common-cu129-ubuntu-2204-nvidia-580`
  - `200 GB pd-balanced`
- VM external IP creation failed because of org policy:
  - `constraints/compute.vmExternalIpAccess`
- Cloud NAT added after first outbound connectivity test failed

## Validation Artifacts

- SSH access path: `gcloud compute ssh --tunnel-through-iap`
- Initial hardware probe:
  - hostname OK
  - `nvidia-smi` showed `NVIDIA L4`
  - `curl https://huggingface.co` timed out before NAT and succeeded after NAT

## Pending Public Ingress

- Public direct VM IP is not allowed in this project
- Public ingress uses:
  - external HTTP load balancer
  - frontend IP `34.49.229.203`
  - backend to the no-address VM
- Smoke validation succeeded through the load balancer for:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Later upgrade path:
  - domain
  - HTTPS target proxy
  - managed or self-managed certificate
