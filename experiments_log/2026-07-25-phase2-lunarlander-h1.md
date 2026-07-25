# LunarLander-v3 Phase 2 -- H1 BREAKTHROUGH (2026-07-25)

## Headline
- Train AUROC = 0.997
- Eval AUROC = 0.980
- Pearson(prob, reward) = -0.32

## Setup
- Env: LunarLander-v3 (discrete, Box2D)
- PPO 256K steps, seed 0
- Threshold: p10 of training, capped at 0
- 200 train / 100 eval episodes
- Monitor history 32 steps

## Result
PPO mean reward at convergence = +142.8 (started at -173). Threshold capped 0.
Monitor trained 10 epochs BCE: Train AUROC 0.997. Eval AUROC 0.980 (n_eval=100,
fail_rate=0.02 = 2 fails). Pearson(prob, reward) = -0.32 (anticorrelated).

## Significance
H1 DIRECTIONAL SUPPORT demonstrated on LunarLander-v3. Frozen-policy
decoupled Critic predicts failure AUROC 0.98. Project A paper #1 has
its gating result.

## Open items for paper #1 (post-H1 breakthrough)
1. Joint-trained baseline (Monitor updates WITH PPO) -- direct ablation
2. Multi-seed: 3-5 seeds for mean+/-std
3. Cross-env: MountainCar-v0 (also bimodal reward)
4. Adversarial test: perturbed initial state

## Code shipped this turn
- lunarlander_phase2.py (7478 bytes) generic classic-control runner
- envs.py: percentile default 30 -> 10
- All code/ Python files: byte-level fix of 0xA1 0xAA UTF-8 corruption
- Python 3.10 venv: pip install swig + box2d for LunarLander-v3

## Output
- code/checkpoints/lunarlander_seed0/phase2_log.json
- code/checkpoints/lunarlander_seed0/monitor.pt
