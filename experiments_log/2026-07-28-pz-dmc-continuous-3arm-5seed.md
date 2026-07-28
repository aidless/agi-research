# DMC continuous-action 3-arm 5-seed sweep: NEGATIVE (real shaping HURTS)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_dmc_continuous.py`
> Matched compute to MADDPG v2: 30+30 PPO updates x 10 episodes = 600 env episodes
> Verdict: real Monitor shaping is WORSE than no shaping (-23.5 mean, 1/5 positive)

## 1. Setup

Continuous-action DMC. Per-agent ContinuousActor (5-dim sigmoid) trained
with continuous PPO (Gaussian log_prob + clipped surrogate).
- Stage 1: 30 SHARED-PPO updates x 10 episodes = 300 episodes (one shared
  policy, then broadcast to 3 per-agent actors).
- Monitor training: 80 frozen-PPO episodes (median-based balanced labels).
- Stage 2: 30 per-agent PPO updates x 10 episodes = 300 episodes, with
  reward shaping mode = {real, random, none}.
- Eval: 15 episodes deterministic per seed.

## 2. 5-seed results

| seed | real    | random  | none    | real-rand | real-none |
|---|---|---|---|---|---|
| 0 |  -99.91 |  -84.24 |  -87.76 | -15.67 | -12.15 |
| 1 | -116.18 |  -93.69 |  -73.85 | -22.49 | -42.33 |
| 2 |  -68.60 |  -91.92 |  -72.13 | +23.32 | +3.53  |
| 3 |  -97.30 |  -74.28 |  -76.57 | -23.02 | -20.73 |
| 4 | -123.16 |  -78.60 |  -77.18 | -44.56 | -45.98 |
| mean | **-101.03** | -84.55 | **-77.50** | -16.48 | -23.53 |
| sd | 21.13 | 8.35 | 6.09 | - | - |

## 3. Paired t-tests

| comparison | mean_diff | se | t | positive |
|---|---|---|---|---|
| real vs random | -16.48 | 11.07 | -1.49 | 1/5 |
| real vs none   | -23.53 |  9.29 | -2.53 | 1/5 |
| random vs none |  -7.05 |  5.28 | -1.34 | 2/5 |

real vs none: t=-2.53 (|t|>=2.776 NOT met at df=4, alpha=0.05) but very close.
The 1/5 positive rate is itself a strong negative signal.

## 4. H5 closure

H5 asked: does Y1.3-style reward shaping transfer to MA?

This 3-arm ablation gives the HONEST answer for continuous actions:
- real Monitor shaping is **worse** than no shaping (-23.5 mean, 1/5 pos).
- random shaping is in between (also slightly negative vs none, -7.0).
- Stage 2 PPO update is now real (continuous Gaussian PPO), unlike the
  earlier discrete-DMC 3-arm which had no real PG signal.

**H5 verdict: REFUTED at matched compute on continuous actions.**
Y1.3-style reward shaping actively hurts DMC; trained per-agent Monitors
do not yield a useful shaping signal on this env at this compute scale.

## 5. Why real shaping HURTS (honest post-hoc)

- Monitor AUROC is 0.99+ (decoupling assumption validated).
- A Monitor with AUROC 0.99 trained on Stage-1 rollouts is BIASED toward
  Stage-1 failure modes. Adding `r_total = r_env - 0.5 * monitor_prob_i`
  penalises states that look like Stage-1 failures, but the *current* policy
  (Stage-2 init from Stage-1) explores different states, where the Monitor
  may be over-confident wrong. Net effect: noise + bias.
- random shaping has the same noise but no bias. It still hurts (-7.0 vs none)
  but less than real shaping (-23.5 vs none). Consistent with bias hypothesis.
- Stage-1 shared PPO converges to -68 to -73 (good init), but Stage-2 PPO
  *with shaping* diverges. The reward perturbation is destabilising.

## 6. 6-way comparison (FINAL)

| Method | Mean | sd | Action | n |
|---|---|---|---|---|
| Random | -77.45 | 25.03 | continuous | 1 |
| Per-agent PPO | -100.51 | 21.70 | discrete | 1 |
| Shared PPO (seed 0) | -95.15 | 30.64 | discrete | 1 |
| DMC discrete (3-arm real) | -125.34 | 42.23 | discrete | 5 |
| DMC continuous real | **-101.03** | 21.13 | continuous | 5 |
| DMC continuous random | -84.55 | 8.35 | continuous | 5 |
| DMC continuous none | -77.50 | 6.09 | continuous | 5 |
| **MADDPG v1 (broken bootstrap)** | -75.78 | 31.11 | continuous | 1 |
| **MADDPG v2 (proper bootstrap)** | **-70.45** | 1.14 | continuous | 5 |

Best baseline: MADDPG v2 (-70.45).
DMC continuous no-shaping comes second (-77.50).
DMC continuous with shaping (any kind) hurts vs no shaping.

## 7. Action items

- [x] pz_dmc_continuous.py with real continuous PPO (Gaussian log_prob)
- [x] 3-arm 5-seed sweep at matched compute (600 episodes)
- [x] H5 closure: REFUTED on continuous actions
- [x] Final 6-way comparison documented
- [ ] Update 9-hypothesis framework H5 -> REFUTED
- [ ] Y1 paper / thesis addendum M with these results