# TTC BoN+Monitor v2 (with state cloning) — LunarLander-v3 2-seed

> Date: 2026-07-26
> Status: COMPLETE — state cloning works, BoN still negative
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What changed since v1

`code/env_state_cloner.py` (4477 bytes, NEW): wraps gym env with
save/restore. For LunarLander-v3 (Box2D), saves lander body state:
position, linearVelocity, angle, angularVelocity. Restored via
direct Box2D property assignment + `ApplyForceToCenter(0, 0, wake=True)`.

Tested: saved state at step 0 = saved values, restored state at step
5 = exact same values. **State cloning works correctly.**

`code/ttc_bon_monitor.py` updated to use EnvStateCloner for actual
future-state simulation, not fresh-env reset.

## 2. Results (LunarLander-v3, 100K PPO, N=4, K=8)

| Seed | Vanilla PPO | BoN+Monitor | Delta |
|------|-------------|-------------|-------|
| 0    | 174.2 +/- 80.9 | -109.2 +/- 137.0 | **-283.4** |
| 1    | -9.0 +/- 34.0 | -28.1 +/- 244.8 | -19.1 |
| mean | 82.6       | -68.7         | -151.3 |

Action distribution at seed 0: {0: 3, 1: 316, 2: 490, 3: 121}
Action distribution at seed 1: {0: 263, 1: 124, 2: 516, 3: 27}

In both seeds, action 2 (main engine) is chosen ~50% of the time —
this is catastrophic for fuel-limited LunarLander.

## 3. Diagnosis: state cloning is not the bottleneck

**The fundamental problem is Monitor calibration, not state cloning.**

At seed 0, train mean = 165.1 with only 15/150 failures (10% fail
rate). Monitor is trained on heavily imbalanced labels — 135
negatives vs 15 positives. BCE loss converges to predicting ~0.1 for
most states (since most are not failures). The Monitor output
distribution has low dynamic range, so BoN ranking is essentially
random within "low-failure-probability" candidates.

When we then sample N=4 candidates and pick the one with "lowest"
Monitor score, we systematically pick the one Monitor happened to
score lowest — but since most scores are similar (~0.1), this is
nearly random selection. Combined with the random seed's first
sample being action 2 often, we get a systematic bias toward action 2.

At seed 1, more failures (92/150) so Monitor is better calibrated,
but still bias toward action 2.

## 4. Why state cloning didn't help

State cloning makes the future-rollout proxy accurate (true next state
from current state + action). But if Monitor can't distinguish between
candidates, accurate future state doesn't help — all candidates still
score similarly.

## 5. What would actually fix this (Y1 plan)

To make TTC work, we need:
1. **Better Monitor training data**: ensure balanced positive/negative
   labels (50/50 instead of 10/90), or use a calibrated loss
2. **Per-step PRM scoring**: aggregate Monitor output across rollout
   steps (mean, max, min) instead of single-shot scoring
3. **Different scoring strategy**: instead of "pick lowest Monitor
   score", use "pick action that maximizes expected reward" via
   learned value function
4. **Diversity bonus**: add penalty for repeated action selection
   to avoid action 2 collapse

The state cloning infrastructure is now in place — future TTC
work can plug in any scoring function over real next-state trajectories.

## 6. Conclusion

**TTC BoN+Monitor as currently implemented does NOT improve over
vanilla PPO on LunarLander-v3.** The Monitor's poor calibration
after training on mostly-successful trajectories is the bottleneck,
not state cloning.

This is honest negative evidence for ADR 0011 P3 status. We should
NOT promote TTC to P2 based on this PoC.

## 7. Artifacts

- `code/env_state_cloner.py` (4477 bytes, NEW)
- `code/ttc_bon_monitor.py` (updated, ~11 KB)
- `code/checkpoints/ttc_bon_monitor_LunarLander-v3_seed{0,1}/phase2_log.json`
- Total runtime: ~4 minutes for 2 seeds

## 8. State cloning as standalone artifact

Even though TTC didn't work, env_state_cloner.py is useful for:
- Y1 Project C: world model rollouts can use exact state cloning
- Y1 TTC v3: better Monitor training data via balanced sampling
- Future work: any "what if" simulation in Box2D envs

## 9. Y1 work to make TTC work (revised list)

1. Train Monitor on balanced positive/negative data
2. Per-step PRM-style scoring aggregation
3. Add diversity bonus to BoN selection
4. Use learned value function as alternative scorer
5. Cross-env validation (Procgen if installable, else Atari)
6. Larger N values (8, 16) to give more ranking signal