# v8 10K-episode n=50: dlr_only effect shrinks further, NOT sig

> Date: 2026-07-30
> Setup: PettingZoo Simple Spread v3 (continuous, 10K ep/seed, 800 PPO updates x 10 episodes)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v8.py`
> 50 paired seeds per arm (s0-s49)
> Wall time: 2 hours 34 min

## Background

After the 10K n=5 (direction reversed) and 10K n=20 (direction
back to positive but NOT sig) results, we extended to 10K n=50
to test if the dlr_only effect at 10K is real.

## Results (n=50, 10K episodes)

| arm | n | mean | sd |
|---|---|---|---|
| dlr_only | 50 | -73.6567 | 3.9239 |
| no_verifier | 50 | -74.2910 | 4.2847 |

**Paired test (n=50)**:
- mean_diff = +0.6343 (dlr_only is BETTER than no_verifier)
- sd_diffs = 4.9131 (high variance, similar to 10K n=20)
- t = +0.913
- 95% CI = [-0.7275, +1.9962] (INCLUDES 0)
- p_uncorrected = 0.3613
- p_bonferroni (2 tests) = 0.7226
- n_pos = 24/50 (48%)

**Status: NOT statistically significant.** The 95% CI includes 0,
and the effect is barely above zero (0.63 mean vs 4.91 sd).

## FINAL effect-shrinkage trajectory (all 6 measurements)

| sample | mean_diff | t | p | n_pos | sig? |
|---|---|---|---|---|---|
| 800ep n=5 | +0.15 | +0.99 | 0.378 | 3/5 (60%) | NOT sig |
| 800ep n=30 | +0.14 | +3.22 | 0.0013 | 20/30 (67%) | YES |
| **800ep n=100** | **+0.06** | **+2.30** | **0.0216** | **64/100 (64%)** | **YES (Bonf)** |
| 10K n=5 | -1.69 | -0.998 | 0.378 | 2/5 (40%) | NOT sig (NOISE) |
| 10K n=20 | +1.27 | +1.44 | 0.151 | 10/20 (50%) | NOT sig |
| **10K n=50** | **+0.63** | **+0.91** | **0.361** | **24/50 (48%)** | **NOT sig** |

## What this means

### The dlr_only effect at 10K is NOT significant

Across 3 different sample sizes at 10K (n=5, n=20, n=50):
- n=5: -1.69 (NOT sig; probably noise)
- n=20: +1.27 (NOT sig, p=0.15)
- n=50: +0.63 (NOT sig, p=0.36)

All three are NOT statistically significant. The dlr_only effect
at 10K is small (likely between 0 and 1) and uncertain (95% CI
includes 0).

### Comparison: 800ep n=100 vs 10K n=50

| compute | n | mean_diff | t | p_bonf | n_pos | sig? |
|---|---|---|---|---|---|---|
| 800ep | 100 | +0.06 | +2.30 | 0.043 | 64% | **YES** |
| 10K | 50 | +0.63 | +0.91 | 0.723 | 48% | NOT |

**The 800ep result IS significant at n=100.** The 10K result is
NOT significant even at n=50. Why the difference?
1. **The variance is much higher at 10K** (sd_diffs 4.9 vs 0.27)
2. **The effect is similar in absolute terms** (0.06 vs 0.63,
   though 10K is 10x larger)
3. **The signal-to-noise ratio is much WORSE at 10K** (Cohen d_z
   = 0.06/0.27 = 0.22 at 800ep, vs 0.63/4.91 = 0.13 at 10K)

### The honest interpretation

The dlr_only effect at 10K is **NOT robust**:
- Mean is positive (similar direction to 800ep)
- But the effect is small (~0.6) and variance is huge (~4.9)
- 95% CI includes 0, so we cannot rule out effect = 0
- The signal-to-noise ratio is WORSE at 10K than at 800ep
- The 800ep result is the most reliable estimate of the dlr_only
  effect at our compute scale

**We do NOT recommend dlr_only for use with longer training
regimes** (10K+ episodes) without further hyperparameter tuning
of the DLR predicate coefficients.

## Code and data

- Launcher: `experiments_log/_run_v8_10k_n50.ps1`
- Per-seed logs: `experiments_log/_v8_10k_n50_*.log` (100 logs)
- Per-seed checkpoints: `projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v8/seed{0-49}_*`
  (NOTE: these OVERWROTE the 800ep results for s0-s49; the
  800ep s50-s99 are still available at `seed{50-99}_*`)
