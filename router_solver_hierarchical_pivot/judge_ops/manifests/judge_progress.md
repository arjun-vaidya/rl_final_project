# Judge Deployment Progress

- [x] Fill in `env/gcp.env.local`
- [x] Fill in `env/judge.env.local`
- [x] Fill in final `env/local_judge.env.local` public URL
- [x] Create VM and reserve static IP
- [x] Decide whether to keep or replace the unused regional IP
- [x] Create Cloud NAT for egress
- [x] Push `judge_ops` bundle to the VM
- [x] Run remote bootstrap script
- [x] Confirm `vllm-judge.service` is healthy
- [x] Confirm NGINX is healthy
- [x] Create external HTTP(S) load balancer
- [x] Point public URL at the LB frontend
- [x] Confirm public smoke test
- [x] Pass local `smoke_test_remote_judge.sh`
- [x] Launch `G=6` slim run
- [ ] Record first-run metrics and issues
