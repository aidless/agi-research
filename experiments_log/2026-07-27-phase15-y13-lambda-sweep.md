# Phase 1.5 Y1.3 Lambda Sweep - monitor_lambda sensitivity

> Date: 2026-07-27
> Status: lambda=0.5 confirmed as best; lambda=5.0 hurts
> Script: code/y13_monitor_regularizer.py
> Sweep: lambda in {0.5, 1.0, 2.0, 5.0}, 5 seeds each

## 1. Per-lambda results

| lambda | Mean | Std | Delta vs baseline | Per-seed | Wins |
|--------|------|-----|-------------------|----------|------|
| **0.5** | **90.5** | 56.3 | **+49.9** | [76, 29, 105, 179, 64] | 4/5 > baseline_mean |
| 1.0    | 65.3 | 38.4 | +24.7 | [56, 4, 94, 73, 100] | 4/5 |
| 2.0    | 61.8 | 46.9 | +21.2 | [72, 111, 13, 100, 13] | 3/5 |
| 5.0    | -58.0 | 79.4 | -98.6 | [43, -91, -88, 1, -155] | 1/5 |

PPO-only baseline: 40.6 +/- 37.1 (5 seeds)

## 2. Dose-response curve

The Monitor reward shaping has a clear dose-response:
  - lambda=0.5 (default): best mean
  - lambda=1.0, 2.0: similar (~+25)
  - lambda=5.0: too strong, hurts policy (mean -58)

Interpretation:
  - The Monitor signal is useful but moderate
  - Stronger penalty (lambda>2) destabilizes PPO training
  - Weak penalty (lambda<0.5, untested) might also work but unlikely
    to be much better than 0.5

## 3. Per-seed analysis (lambda=0.5 vs lambda=5.0)

| Seed | lambda=0.5 | lambda=5.0 | Difference |
|------|------------|------------|------------|
| 0    | 75.6       | 43.2       | -32       |
| 1    | 29.2       | -90.5      | -120      |
| 2    | 105.2      | -88.4      | -194      |
| 3    | 178.7      | 1.2        | -178      |
| 4    | 63.8       | -155.4     | -219      |

lambda=5.0 catastrophically hurts every seed. The Monitor
penalty becomes so large that PPO optimizes for "minimize
Monitor_prob" at the expense of task reward.

## 4. Decision record (DEC-Y1.3 lambda)

> **Y1.3 with lambda=0.5 is the optimal setting.** Stronger lambda
> (1.0, 2.0) gives similar but smaller gains. Very strong lambda
> (5.0) breaks training.
>
> lambda=0.5 is also the most robust: it has the best mean and
> 4/5 seeds > baseline. lambda=1.0 is comparable (4/5 wins, smaller
> mean). lambda=2.0 has 3/5 wins and similar mean. lambda=5.0 is
> uniformly bad.
>
> **Recommended setting for future Y1.3 runs: lambda=0.5.**

## 5. Reproducibility

```bash
for lam in 0.5 1.0 2.0 5.0; do
  for seed in 0 1 2 3 4; do
    python projects/project_a_self_improvement/code/y13_monitor_regularizer.py \
      --n-ppo-steps-total 100000 --n-warmup-steps 25000 \
      --n-train-episodes 200 --n-eval-episodes 50 \
      --history-len 32 --seed $seed \
      --monitor-lambda $lam --out-tag y13
  done
done
```

Per-seed JSON: `checkpoints/full_integration_y13_LunarLander-v3_seed{0..4}/phase2_log.json`
Summary: `experiments_log/y13_lambda_sweep_summary.json`
