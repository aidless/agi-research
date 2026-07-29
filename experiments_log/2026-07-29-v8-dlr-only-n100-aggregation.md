# v8 dlr_only n=100 aggregation: effect confirmed but smaller than n=30 estimate

> Date: 2026-07-29
> Setup: PettingZoo Simple Spread v3 (continuous, 800 ep/seed)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v8.py`
> 100 paired seeds per arm (extends existing n=30 to n=100)

## Background

The v8 dlr_only n=30 result (+0.1447, p<0.005, t=+3.216, 20/30
positive) was a statistically significant positive result. But
n=30 is small for estimating the true effect size. We extended
to n=100 (70 new seeds s30-s99) to confirm the effect at higher
statistical power.

## Setup

- 140 new jobs: 70 dlr_only + 70 no_verifier (s30-s99)
- 4-parallel execution
- Started: 2026-07-29 21:12:28
- Completed: 2026-07-29 22:22:48
- Wall time: 70 min 20 sec

## Results (n=100)

| arm | n | mean | sd |
|---|---|---|---|
| dlr_only (DLR in critic) | 100 | -69.5913 | 1.7899 |
| no_verifier (v2 baseline) | 100 (paired) | -69.6530 | 1.8378 |

**Paired test (n=100)**:
- mean_diff = +0.0617
- sd_diffs = 0.2685
- t = +2.297
- 95% CI = [+0.0084, +0.1149]
- p_uncorrected (two-sided) = 0.0216
- p_bonferroni (2 tests) = 0.0433
- n_pos = 64/100 (64%)

**STATISTICALLY SIGNIFICANT** at p<0.05 even with Bonferroni
correction. The 95% CI excludes 0.

## Effect-shrinkage trajectory

| sample | mean_diff | t | p_uncorr | p_bonf | n_pos | sig? |
|---|---|---|---|---|---|---|
| n=5 | +0.1500 | +0.99 | 0.3781 | 0.7562 | 3/5 (60%) | NOT sig |
| n=30 | +0.1447 | +3.216 | 0.0013 | 0.0026 | 20/30 (67%) | **YES** |
| **n=100** | **+0.0617** | **+2.297** | **0.0216** | **0.0433** | **64/100 (64%)** | **YES** |

The effect SHRANK from +0.1447 (n=30) to +0.0617 (n=100) -- about
half. This is the textbook signature of a small effect that gets
more precisely estimated with larger samples.

The n=100 estimate of +0.0617 is probably closer to the true
effect. The earlier n=30 estimate of +0.1447 was inflated by
small-sample variability.

## Practical implications

- **Effect is REAL and STATISTICALLY SIGNIFICANT** at n=100
  (p<0.05 with Bonferroni)
- **Effect is SMALL**: +0.0617 mean on a baseline of -69.65, a
  relative improvement of ~0.09%
- **Effect is STABLE** in direction (64/100 positive)
- **Effect is HONESTLY small** -- this is the n=100 estimate, not
  inflated by small-sample noise

The n=100 result is the most reliable estimate of the dlr_only
effect. Earlier n=5 and n=30 estimates were likely upward
biased.

## What this means for the 6-pathway story

1. **dlr_only is the only publishable result** (n=100 confirms)
2. **The effect is smaller than initially reported** (textbook
   small effect at higher statistical power)
3. **The Monitor signal still does not transfer** (v3, v4, v5,
   v6, v7 all REFUTED)
4. **The trust head still ignores its input** (v6 n=30 CLEAN
   bit-for-bit identity stands)

## Updated paper text (Section 4.2)

**v8 dlr_only at n=100**: mean_diff = +0.0617, t = +2.297,
p_uncorr = 0.0216, p_bonf (2 tests) = 0.0433, 95% CI
[+0.0084, +0.1149], n_pos = 64/100 (64%). The effect is
STATISTICALLY SIGNIFICANT (even with Bonferroni correction) but
SMALLER than the n=30 estimate (+0.1447). The shrinkage from
n=30 to n=100 is the textbook signature of a small effect that
gets more precisely estimated with larger samples.

Cohen d_z = mean_diff / sd_diffs = 0.0617 / 0.2685 = 0.23
(small-to-medium effect by Cohen's convention; on a metric
where baseline = -69.65, the relative improvement is ~0.09%).

## What this changes

- The dlr_only effect is REAL and statistically significant, but
  SMALLER than initially reported
- The honest effect size estimate is +0.06 (n=100), not +0.14
  (n=30)
- The paper should be transparent about the effect-shrinkage
  trajectory
- The publishability is preserved (still p<0.05 with Bonferroni)
  but the practical impact is small (~0.09% relative improvement)

## Code and data

- Launch script: `experiments_log/_run_v8_n100.ps1`
- Aggregation: this file
- Per-seed data: `projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v8/`
