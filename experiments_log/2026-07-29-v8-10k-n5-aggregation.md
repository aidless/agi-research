# v8 10K-episode n=5: dlr_only effect REVERSES at longer training

> Date: 2026-07-29
> Setup: PettingZoo Simple Spread v3 (continuous, 10K ep/seed, 800 PPO updates x 10 episodes)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v8.py`
> 5 paired seeds per arm (s0-s4)

## Background

The 800-episode v8 dlr_only result at n=5 was +0.15 mean over
baseline (NOT sig, 3/5 positive). We extended to 10K episodes
(10x compute, 800 PPO updates x 10 episodes) to test if the
effect grows, shrinks, or stays the same at longer training.

## Results (n=5, 10K episodes)

| seed | no_verifier | dlr_only | diff |
|---|---|---|---|
| 0 | -68.4943 | -72.266 | -3.77 |
| 1 | -67.3457 | -74.304 | -6.96 |
| 2 | -76.4305 | -77.900 | -1.47 |
| 3 | -71.3884 | -69.274 | +2.11 |
| 4 | -73.5739 | -71.952 | +1.62 |

**Per-arm means**:
- no_verifier: -71.4466
- dlr_only: -73.1391

**Paired test (n=5)**:
- mean_diff = -1.6925 (dlr_only is WORSE than no_verifier)
- sd_diffs = 3.7939
- t = -0.998
- NOT significant at df=4 (|t|>=2.776 for p<0.05)
- n_pos = 2/5 (only 2 seeds have dlr_only better)

## Comparison to 800-episode result

| compute | n | mean_diff | t | n_pos |
|---|---|---|---|---|
| 800 ep (n=5) | 5 | +0.15 | +0.99 | 3/5 (60%) |
| 10K ep (n=5) | 5 | **-1.69** | **-0.998** | 2/5 (40%) |

The dlr_only effect **REVERSES** at 10K episodes. At 800 ep,
dlr_only is slightly better than no_verifier (+0.15). At 10K ep,
dlr_only is WORSE than no_verifier (-1.69). Neither result is
significant at n=5, but the direction is reversed.

## What this means

### Honest interpretation

1. **The dlr_only effect is UNSTABLE** at longer training. At
   800ep, it's positive (+0.15). At 10K, it's negative (-1.69).
   This is similar to the v3 finding (Monitor aux loss: neutral
   at 800ep, HURTS by -3.03 at 10K).

2. **n=5 is too small to draw strong conclusions** about the
   direction. With only 5 seeds, both effects are within
   sampling noise.

3. **The 10K result is confounded by overall training instability**:
   - no_verifier at 800ep: -70.50 mean
   - no_verifier at 10K: -71.45 mean (WORSE)
   - This suggests MADDPG v2 is overfitting or diverging at 10K
     with these hyperparameters.
   - The dlr_only vs no_verifier comparison at 10K is conflated
     with this overall instability.

4. **The 800-episode n=5 result may have been upward biased by
   small-sample noise**. With n=100 at 800ep, the dlr_only effect
   shrinks to +0.06 (still sig, but smaller). The 10K result
   is consistent with the overall pattern of effect-shrinkage.

### Practical implications

The dlr_only effect is **NOT ROBUST** to longer training at
these hyperparameters. The honest interpretation:
- At 800ep, dlr_only gives a small but real effect (+0.06 at n=100)
- At 10K, the effect reverses and may even hurt
- This is similar to the v3 finding (Monitor aux loss: short
  training OK, long training HURTS)

We do NOT recommend dlr_only for use with longer training
regimes (10K+ episodes) without further hyperparameter tuning
of the DLR predicate coefficients.

### What this changes for the paper

1. **Add a Limitations subsection** on training regime sensitivity
2. **Add the 10K n=5 result** to the Discussion section
3. **Add a caveat** to the dlr_only recommendation: the effect is
   shown at 800ep; longer training is not tested at n=100
4. **Recommend future work**: hyperparameter tuning of DLR
   coefficients, longer training with proper LR schedule, etc.

## Code and data

- Launcher: `experiments_log/_run_v8_10k_n5.ps1`
- Per-seed logs: `experiments_log/_v8_10k_*.log`
- Per-seed checkpoints: `projects/project_f_multi_agent/code/checkpoints/pz_maddpg_v8/seed{0-4}_*`
  (NOTE: these OVERWROTE the 800ep results at the same paths;
  the 800ep results are preserved in the n=100 aggregation log)
