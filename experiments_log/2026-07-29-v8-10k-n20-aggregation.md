# v8 10K-episode n=20: dlr_only effect is larger but noisier (NOT sig at n=20)

> Date: 2026-07-29 to 2026-07-30
> Setup: PettingZoo Simple Spread v3 (continuous, 10K ep/seed, 800 PPO updates x 10 episodes)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v8.py`
> 20 paired seeds per arm (s0-s19)

## Background

The 10K-episode v8 dlr_only result at n=5 was -1.69 (NOT sig,
direction REVERSED from 800ep). We extended to n=20 to test if
the reversal was real or just n=5 noise.

## Results (n=20, 10K episodes)

| arm | n | mean | sd |
|---|---|---|---|
| dlr_only | 20 | -72.7442 | 2.3780 |
| no_verifier | 20 | -74.0181 | 3.7930 |

**Paired test (n=20)**:
- mean_diff = +1.2738 (dlr_only is BETTER than no_verifier)
- sd_diffs = 3.9655 (much higher than 800ep)
- t = +1.437
- 95% CI = [-0.4641, +3.0118] (INCLUDES 0)
- p_uncorrected = 0.1508
- p_bonferroni (2 tests) = 0.3017
- n_pos = 10/20 (50%)

**Status: NOT statistically significant.** The 95% CI includes 0,
so we cannot reject H0: effect = 0 at n=20.

## Effect-shrinkage trajectory (all 800ep and 10K results)

| sample | mean_diff | t | p | n_pos | sig? |
|---|---|---|---|---|---|
| 800ep n=5 | +0.15 | +0.99 | 0.378 | 3/5 (60%) | NOT sig |
| 800ep n=30 | +0.14 | +3.22 | 0.0013 | 20/30 (67%) | **YES** |
| 800ep n=100 | +0.06 | +2.30 | 0.0216 | 64/100 (64%) | **YES (Bonf)** |
| 10K n=5 | -1.69 | -0.998 | 0.378 | 2/5 (40%) | NOT sig |
| **10K n=20** | **+1.27** | **+1.44** | **0.151** | **10/20 (50%)** | **NOT sig** |

## What this means

### The 10K n=5 result was noise

The 10K n=5 result of -1.69 (direction REVERSED) is likely just
n=5 noise. The 10K n=20 result of +1.27 (direction SAME as 800ep)
suggests the dlr_only effect is not reversed at 10K.

### The dlr_only effect MAY be larger at 10K, but the variance is also much larger

| sample | mean_diff | sd_diffs | ratio |
|---|---|---|---|
| 800ep n=100 | +0.06 | 0.27 | 1x |
| 10K n=20 | +1.27 | 3.97 | 21x mean, 14x sd |

The 10K effect has 21x the mean AND 14x the variance. The signal-
to-noise ratio (Cohen d_z) is similar:
- 800ep n=100: d_z = 0.06 / 0.27 = 0.22
- 10K n=20: d_z = 1.27 / 3.97 = 0.32

So the 10K effect has slightly better signal-to-noise, but the
absolute effect is also much larger (and noisier).

### Why is the variance so much higher at 10K?

This is likely because:
1. **MADDPG v2 is unstable at 10K with these hyperparameters**
   (no_verifier at 10K = -74.0 vs -69.7 at 800ep, much worse)
2. **The training variance across seeds is much larger at 10K**
   because different seeds converge to very different points
3. **The dlr_only effect may be sensitive to where in the
   training trajectory the model is when the critic input is
   used**

The honest interpretation: **at 10K, the dlr_only effect is not
robust**. The mean is positive (similar direction to 800ep) but
the variance is so large that we cannot conclude the effect is
real at 10K.

### Practical implications

1. **At 800ep (80 PPO updates), dlr_only is a small but real
   positive effect** (+0.06 at n=100, p<0.05 with Bonferroni)
2. **At 10K (800 PPO updates), dlr_only MAY be larger in absolute
   terms but is not statistically significant at n=20**
3. **We do NOT recommend dlr_only for use with longer training
   regimes** without further hyperparameter tuning
4. **The 800ep result is the most reliable estimate of the
   dlr_only effect at our compute scale**

### Code and data

- Launcher: `experiments_log/_run_v8_10k_n20.ps1`
- Per-seed logs: `experiments_log/_v8_10k_n20_*.log`
- Per-seed checkpoints: `projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v8/seed{0-19}_*`
  (NOTE: these OVERWROTE the 800ep results for s0-s19; the
  800ep s20-s99 are still available)
