# Router-Solver Training Quick Start (10 PM Run)

## Prerequisites
- SSH access to rlfp-g2 VM (configured in ~/.ssh/config)
- WANDB_API_KEY in .env file
- Latest code pulled from main branch

## Steps to Run Training

1. **SSH into VM**
   ```bash
   ssh rlfp-g2
   ```

2. **Navigate to project**
   ```bash
   cd /home/vaidya/router_solver
   ```

3. **Pull latest changes**
   ```bash
   git pull
   ```

4. **Kill any existing processes**
   ```bash
   pkill -9 python3
   sleep 3
   ```

5. **Clear old logs**
   ```bash
   rm -f logs/train_router_solver.log
   ```

6. **Start training with nohup**
   ```bash
   PYTHONPATH=. nohup python3 -u src/training/train_router_solver.py --config configs/router_solver.yaml > logs/train_router_solver.log 2>&1 &
   ```

7. **Monitor progress**
   ```bash
   tail -f logs/train_router_solver.log
   ```
   - Watch for "Generating X rollouts" messages
   - tqdm progress bar shows estimated time remaining
   - Ctrl+C to exit tail (training continues)

8. **Check W&B dashboard** (optional)
   - https://wandb.ai/av3315-columbia-university/router-solver
   - Shows loss, accuracy, and other metrics in real-time

## Current Configuration
- **Batch size:** 1 question per step
- **Rollouts:** 2 per question (total 2 per step)
- **Steps:** 500
- **Est. time:** ~7 hours
- **GPU:** L4 (24GB)

## If Training Crashes
- Check logs: `tail -200 logs/train_router_solver.log`
- Look for CUDA out of memory errors
- If OOM: reduce batch_size or group_size in configs/router_solver.yaml
- Restart from step 4 above

## After Training Completes
- Model saved to: `experiments/router_solver_decomposed/final_hierarchical_model`
- Checkpoints every 50 steps in same directory
- Metrics logged to W&B project
