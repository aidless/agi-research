# Acrobot-v1 Phase 2 (C attempt) - (2026-07-25)

## Setup
- Env: Acrobot-v1 (3 actions, 6-dim obs)
- PPO 256K, seed 0
- Generic classic_phase2.py: --threshold-floor=-1e9 --percentile 10.0
- 200 train / 100 eval episodes

## Result
- PPO mean reward: -90 (started at -400, converged to -80)
- Dynamic threshold: max(-1e9, p10) = -112
- BUT is_failure_episode uses STATIC FAILURE_THRESHOLDS (Acrobot=-500).
  With mean -80, all train labels = "success" (>= -500). Monitor training SKIPPED.
- Eval: 1 of 100 episodes labelled as failure (by static threshold)
- Monitor weights = random init (training was skipped).
- Eval AUROC = 0.697 — but with 0-fail training, this is signal-from-noise
  (random init on 1 positive class), NOT a real result.

## Bug to fix
is_failure_episode in monitor.py uses static FAILURE_THRESHOLDS;
should accept a dynamic threshold from Phase 2 pipeline.

## Significance
LunarLander-v3 Eval AUROC = 0.98 still stands. Acrobot result is
CONTAMINATED by the threshold bug. Fix + retry before claiming cross-env.

## Code shipped
- classic_phase2.py: generalised to take --env flag (env-agnostic runner)
- Added --threshold-floor flag for failure-threshold lower bound
- Fixed args.env scope bug (collect_rollouts now takes env_name param)

## Open: fix the threshold-passing bug
Need to modify FailureDataset to accept threshold parameter, and
classic_phase2.py to pass it through.
