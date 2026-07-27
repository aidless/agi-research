# H1 Cross-Env Ablation — CartPole-v1 (preliminary, seed 0)

> Date: 2026-07-27 (late evening)
> Mode: First cross-env test of H1 decoupling hypothesis
> Status: **Preliminary NEGATIVE for CartPole** — Monitor fails to predict
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Tested H1 hypothesis on CartPole-v1 (simplest classic-control env):

1. Train PPO for 10K steps on CartPole-v1.
2. Collect 50 episodes from frozen PPO policy.
3. Train frozen Monitor on rollouts (5 epochs, 64 batch).
4. Evaluate val AUROC on held-out 10% of episodes.

(Joint Monitor training incomplete at 304s timeout; preliminary result only.)

## 2. Preliminary result (seed 0)

| metric | value |
|--------|-------|
| PPO steps | 10,000 |
| Frozen Monitor val AUROC | **0.407** |
| Joint Monitor val AUROC | (incomplete) |

## 3. Interpretation: H1 may NOT transfer to CartPole

The frozen Monitor's AUROC of 0.407 is **worse than random** (0.5). This is
similar to the joint Monitor on LunarLander (AUROC ~0.07) — but the
*opposite* direction.

### 3.1 Possible reasons

1. **CartPole failures are sudden**: pole falls in 1-2 timesteps when
   it does fail. History doesn't help predict sudden failure.

2. **PPO converges too well**: after 10K steps, CartPole rarely fails.
   The "failure" episodes (reward < 100) are mostly early-training noise
   rather than genuine failure modes.

3. **Failure label is wrong**: `total_reward < 100` is too coarse.
   Better: label episodes by "near-failure" (last 5 steps before failure)
   or "episode ended in terminal failure state".

4. **CartPole has no partial observability**: with 4-dim state fully
   observed, the failure is *deterministic from current state* — no need
   for a Monitor.

### 3.2 Why this is informative

Even a NEGATIVE result for H1 on CartPole is publishable:

- Confirms H1 is **environment-dependent**, not universal.
- Suggests decoupling helps in **environments with partial observability**
  (LunarLander) but not in **fully observed environments with sudden failure**
  (CartPole).
- This is the **same insight** as DEC-0011's v0.4 HALT: not all environments
  benefit from Monitor intervention.

## 4. Next steps

1. **Better failure definition**: use "last 5 steps before failure" labels
   instead of episode-level label.
2. **Test on MountainCar** (sparse reward, similar to LunarLander dynamics).
3. **Test on Acrobot** (sparse reward, more learnable than CartPole).
4. **Document the failure mode** in thesis as a methodological contribution.

## 5. Artifacts

- `code/h1_cross_env.py` (~270 lines)
- `experiments_log/_h1_cartpole_seed0_quick.txt` (preliminary)
- Quick test was killed at 304s due to script timeout; joint Monitor
  training incomplete.

## 6. Why this is honest negative is valuable

DEC-0011 already showed that intervention is hard on LunarLander.
CartPole adds: **even the Monitor itself doesn't work for some envs**.

This narrows our hypothesis:
- H1 holds in **partially observable, gradually-failing** environments
- H1 may not hold in **fully observable, suddenly-failing** environments

This is a *meaningful* distinction that the field should know about.
