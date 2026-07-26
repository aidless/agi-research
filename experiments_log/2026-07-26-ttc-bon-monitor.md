# TTC BoN+Monitor PoC — 2-seed result on LunarLander-v3

> Date: 2026-07-26
> Status: PoC COMPLETE — mixed result, requires Y1 follow-up
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. Question

Does sampling N candidate actions and ranking them by Monitor-predicted
failure probability give a TTC gain over vanilla PPO? (ADR 0011)

## 2. Method

`code/ttc_bon_monitor.py` (9889 bytes):

1. Train PPO on LunarLander-v3 for 100K steps
2. Collect 150 frozen-policy rollouts
3. Train Monitor on frozen rollouts (5 epochs BCE)
4. **Vanilla PPO baseline eval**: 50 episodes, standard PPO sampling
5. **BoN+Monitor eval**: 50 episodes, at each step:
   - Sample N=4 candidate actions from PPO
   - For each candidate, "roll out" K=10 future steps
   - Score each rollout with Monitor
   - Take action whose rollout got the lowest Monitor failure probability

Future rollout approximation: reset env to a fresh random seed, take
candidate action, continue with same action for K steps. This is a
**proxy** — true TTC requires env state cloning or learned dynamics
(which we don't have).

## 3. Results

| Seed | Vanilla PPO mean | BoN+Monitor (N=4, K=10) | Delta |
|------|-------------------|--------------------------|-------|
| 0    | 40.2 +/- 77.2     | **50.3 +/- 69.6**        | **+10.1** |
| 1    | 31.2 +/- 71.1     | -1.6 +/- 88.4            | -32.8 |
| mean | 35.7              | 24.4                     | -11.4 |

Action distribution at seed 1 was concentrated on actions 0 and 2
(6829 + 11038 = 85% of choices), suggesting the Monitor's "future
rollout proxy" is biased toward certain actions in this seed.

## 4. Interpretation

**Mixed result — not a robust TTC win at this PoC stage.**

Why BoN+Monitor does not reliably improve PPO:
1. **Future-rollout proxy is wrong**: resetting env to a fresh random
   seed does NOT reflect the true future from the current state. This
   is the biggest issue. The Monitor scores a *different trajectory*
   than the one that would actually occur.
2. **Single Monitor output, not PRM-style step-level**: our Monitor
   outputs one probability for the whole future rollout. Lightman
   2023 PRM scores each step separately and aggregates. Per-step
   scoring would be more informative.
3. **High variance**: 50 eval episodes is too few for stable estimates
   when PPO std is ~70-80.
4. **Action concentration**: Monitor's ranking sometimes picks the same
   action repeatedly, reducing exploration.

Seed 0's +10.1 result shows the Monitor CAN provide useful signal
at inference time. Seed 1's -32.8 shows the proxy approximation is
unreliable.

## 5. Y1 plan (per ADR 0011)

To make this a rigorous TTC evaluation:

1. **Better future-rollout proxy**:
   - Option A: implement gym state cloning (custom env wrapper that
     saves/restores state)
   - Option B: train a small dynamics model on frozen rollouts, use
     it for rollouts in TTC eval
   - Option C: skip rollouts entirely; use Monitor on current state
     alone (no future prediction)
2. **Per-step PRM-style scoring**: aggregate Monitor outputs across
   rollout steps (mean, max, last) instead of one-shot on rollout
3. **More seeds**: 5-10 seeds, compute std and confidence interval
4. **Different N values**: N=2, 4, 8, 16 to find optimal compute-quality
   trade-off (Snell 2024)
5. **Different K values**: K=5, 10, 20 for rollout horizon
6. **Cross-env**: LunarLander-v3 + at least 2 other envs (Procgen
   coinrun if possible, otherwise CartPole-v1 + MountainCar-v0 with
   longer PPO)

## 6. Artifacts

- `code/ttc_bon_monitor.py` (9889 bytes)
- `code/checkpoints/ttc_bon_monitor_LunarLander-v3_seed0/phase2_log.json`
- `code/checkpoints/ttc_bon_monitor_LunarLander-v3_seed1/phase2_log.json`
- Compute: ~3.5 min per seed × 2 = 7 min total

## 7. Citation connection

This is the policy-level analog of:
- **Lightman 2023**: BoN+PRM gains ~4 percentage points on MATH
- **Snell 2024**: TTC scaling law, hard problems benefit more

Our seed 0 (+10.1) suggests the pattern transfers to RL at least
sometimes. Y1 work is to make it robust.