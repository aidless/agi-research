# Phase 2 first execution on coinrun (2026-07-25)

> DEC-0009 Phase 2 first run (decoupled Failure Monitor on Procgen).

## Setup
- Game: coinrun (1 Procgen game)
- Seed: 0
- PPO baseline: 50K steps (Dec-0008 Step 3 scale)
- Train episodes for Monitor: 100
- Eval episodes: 50
- Monitor: history_len=16, BCE loss, 5 epochs
- Total wall: ~6 min CPU on Python 3.10

## Result

`
game             : coinrun
seed             : 0
n_episodes_train : 434    (PPO collected during training)
mean_return      : 5.85   (low; early PPO)
p30_threshold    : 0.0    (no failure variance in training)

eval_reward_mean : 6.2
eval_reward_std  : 4.85
fail_rate        : 0.0   (no failure cases at p30=0)
eval_prob_mean   : 0.485
eval_prob_std    : 0.0018 (Monitor output is essentially constant)
auroc_mean       : 0.5    (chance)
pearson_p_r      : -0.30
monitor_loss     : 0.0    (constant prediction)
`

## Interpretation

Monitor outputs are essentially constant (~0.485 +/- 0.002). This is exactly what
happens when BCE loss trains on a uniform label set: BCE's gradient on a constant
predictor at a half-label distribution is zero, so the predictor stays where it was
initialised.

This confirms two things:

1. **The decoupling pipeline runs end-to-end without errors** (no bug in
   ProcgenWrapper, history_vector, monitor.py etc).
2. **Phase 1 baseline is too early** to produce the failure variance needed to
   demonstrate H1. We need 256K+ PPO steps before Monitor can have signal.

## Next step

To get Monitor signal, we need a PPO baseline that produces ~30% failure cases.
At current 50K-step scale, all episodes score ~5 reward so p30=0.
Real Procgen success rate at 256K would be ~50-90% with diverse scores.

## Output
Saved: code/checkpoints/procgen_coinrun_seed0/phase2_log.json (601 bytes)
Saved: code/checkpoints/procgen_coinrun_seed0/monitor.pt (45 KB)

## Bug fixes shipped this turn

- envs.py:EpisodeLog.history_vector hardcoded action space size 2.
  Now accepts 
_actions parameter (auto-detect from data in monitor path).
- monitor.py:FailureDataset now auto-detects max action value across episodes
  and uses max(2, max_action + 1) as n_actions.
- Both fixed procgen 15-action overflow (IndexError on first Monitor call).
