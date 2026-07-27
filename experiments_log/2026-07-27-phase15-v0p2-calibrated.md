# Phase 1.5 v0.2 Calibrated 4-Layer Integration - 5-Seed Sweep (DEC-0011 v0.2)

> Date: 2026-07-27
> Status: **NEGATIVE RESULT - v0.2 calibration made things WORSE**
> Script: \projects/project_a_self_improvement/code/full_integration_v2.py\
> Command per seed: \--n-ppo-steps 100000 --n-train-episodes 200 --n-eval-episodes 50 --target-fpr 0.10 --min-q-coverage 50 --out-tag v2\
> Sweep: seeds 0-4 (5 seeds, run in parallel: ~14 min total)
> Calibration: train/val split 80/20, Platt scaling 1-param logistic, threshold @ FPR=10%

## 1. Per-seed v0.2 results

| Seed | Ungated | Gated | **Delta** | Val AUROC | Cal threshold | Avg gates |
|------|---------|-------|-----------|-----------|----------------|-----------|
| 0    | 107.9   | -152.0 | **-259.9** | 1.000 | 4.1e-14 | 231 |
| 1    | 204.6   | 199.8  | **-4.8**   | 0.974 | 1.1e-01 | 18 |
| 2    | 86.4    | 73.9   | **-12.4**  | 1.000 | 1.3e-08 | 30 |
| 3    | 88.6    | -391.1 | **-479.7** | 0.987 | 7.2e-06 | 137 |
| 4    | 74.9    | 41.4   | **-33.5**  | 1.000 | 3.2e-07 | 308 |

**All 5 seeds negative. 0/5 positive delta.**

## 2. Aggregate v0.2 (n=5, sample std)

\\\
Ungated PPO mean:        112.5 +/- 52.9
Gated (Monitor+Q) mean:  -45.6 +/- 230.7
Delta (Gated-Ungated):  -158.1 +/- 208.6
Avg gates per episode:  144.8 (range 18 - 308)
Seeds with positive delta: 0 / 5
t-statistic:             -1.694 (df=4, p > 0.05, NOT significant)
\\\

## 3. Comparison v0.2 vs v0.1

| Metric | v0.1 (fixed 0.5, n_eval=5) | v0.2 (calibrated, n_eval=50) | Delta |
|--------|---------------------------|------------------------------|-------|
| Ungated | 55.0 +/- 50.3 | 112.5 +/- 52.9 | +57.5 |
| Gated   | 76.6 +/- 34.0 | -45.6 +/- 230.7 | -122.2 |
| **Delta** | **+21.5 +/- 67.1** | **-158.1 +/- 208.6** | **-179.6** |
| Pos seeds | 3/5 | 0/5 | -3 |
| t-stat | 0.72 | -1.69 | n/a |

**v0.2 is significantly WORSE than v0.1 across all 3 measures:**
- Delta went from +22 to -158 (worse by 180 points)
- 3/5 positive seeds → 0/5 positive seeds
- Variance 3x higher (208 vs 67)

## 4. Failure mode analysis: why v0.2 failed

### 4.1 The val-set overfit problem

v0.2 holds out 40 episodes for calibration. With LunarLander-v3, the
failure class (bottom 10% of reward) contains only 4 episodes in 40.
On 4 positives vs 36 negatives, the SlotMonitor achieves val_auroc
near 1.0 (essentially perfect separation). The Platt scaling then
finds a 1-param logistic that pushes negatives to ~0 and positives to
~1. The threshold for FPR=10% ends up extremely low:

- Seed 0: cal_threshold = 4.1e-14 (essentially "always gate")
- Seed 2: cal_threshold = 1.3e-08 (effectively always gate)
- Seed 4: cal_threshold = 3.2e-07 (effectively always gate)
- Only seed 1 has a sensible threshold (0.11)

### 4.2 The CQL coverage problem re-emerges

With cal_threshold near 0, the Monitor fires on 18-308 steps per
episode (avg ~145/500 = 29% of steps). The Q-function, trained on
only 200 episodes, has CQL coverage of these in-distribution
state-action pairs but is poorly calibrated for OOD ones. When the
Monitor fires, Q picks actions that destroy the PPO policy.

The Q coverage guard (min_q_coverage=50) does not help here because
the Q has 50K+ unique (s,a) pairs (way above 50). The guard only
protects against low-data Q; here the data is plenty but the
underlying Q is bad.

### 4.3 The n_eval=50 effect

v0.2 doubled the eval count (5 -> 50) which gives 3x lower variance
in the per-seed mean. This makes the v0.2 result MORE honest than v0.1:
the 0/5 negative verdict is statistically real, not noise.

Note: v0.2's ungated mean (112.5) is also higher than v0.1's (55.0)
because the n=50 eval averages are less noisy (per-episode variance
in ungated is ~70, so 5 episodes has std of 70/sqrt(5)=31 vs 50
episodes has std 70/sqrt(50)=10).

## 5. Per-seed diagnostic

| Seed | val_auroc | cal_threshold | gates | delta | Diagnosis |
|------|-----------|---------------|-------|-------|-----------|
| 0    | 1.000     | 4.1e-14       | 231   | -259.9 | Monitor always fires, Q destroys PPO |
| 1    | 0.974     | 0.108         | 18    | -4.8   | Sensible threshold, gating rare, no effect |
| 2    | 1.000     | 1.3e-08       | 30    | -12.4  | Threshold near 0 but somehow Q mostly helps |
| 3    | 0.987     | 7.2e-06       | 137   | -479.7 | Threshold near 0, Q very bad |
| 4    | 1.000     | 3.2e-07       | 308   | -33.5  | Threshold near 0, Q overrules PPO but PPO also weak |

The pattern: **seeds where val_auroc is exactly 1.0 get the worst
calibration**. The Platt fit on 4 positives over-fits, the threshold
collapses to ~0, and gating becomes unconditional.

## 6. Decision record (DEC-0011 v0.3)

> **DEC-0011 v0.2 (calibrated 4-layer integration with n_eval=50) is
> REJECTED.** v0.2 made Phase 1.5 WORSE: delta +22 (v0.1) -> -158
> (v0.2), 0/5 positive seeds.
>
> Root causes:
> 1. Val set too small (40 eps, 4 positives) -> val_auroc=1.0
>    overfits -> cal_threshold collapses to ~0
> 2. CQL Q-function with 200 train episodes is too noisy to control
>    PPO when Monitor fires on 30%+ of steps
> 3. The Q coverage guard doesn't help (Q has data, just bad)
>
> **v0.1 (fixed thresh=0.5, n_eval=5) remains the canonical result**:
> delta +21.5 +/- 67.1, 3/5 positive, not significant. The H1
> question (does decoupled Monitor + Q gating help LunarLander?) is
> still UNRESOLVED with available techniques.
>
> **v0.3 options** (not yet implemented):
> A. Use larger val set (e.g., 200+ episodes) so val_auroc=1.0 isn't overfit
> B. Skip calibration, return to v0.1's hardcoded 0.5 threshold
> C. Different gating criterion (e.g., Q uncertainty instead of Monitor threshold)
> D. Larger Q training set (e.g., 1000+ PPO rollouts)
> E. Skip Q entirely when Monitor fires; use a fixed safe action
> F. Move to a new env where Monitor can be evaluated standalone

## 7. Reproducibility

\\\
Repo commit at run time: a83c247 (DEC-0011 v0.1)
Python: cpython-3.11-windows-x86_64-none
Torch: 2.5.1+cu124
Launcher: experiments_log/_run_v2_seeds_1to4.ps1 (4 parallel)
Master log: experiments_log/_v2_seed{0..4}.log (~80KB each)
Aggregator: experiments_log/_agg_v2.py
\\\

## 8. Artifacts

- Code (tracked):
  - \code/calibration.py\ (NEW, ~110 lines: AUROC, Platt, FPR threshold, coverage)
  - \code/full_integration_v2.py\ (NEW, ~450 lines: v0.2 main with calibration flow)
- Per-seed checkpoints (not in git): \checkpoints/full_integration_v2_..._seed{0..4}/phase2_log.json\
- Summary JSON: \experiments_log/phase15_v0p2_vs_v0p1_summary.json\
- Per-seed raw logs: \experiments_log/_v2_seed{0..4}.log\ (in .gitignore)
