# Phase 1.5 v0.3 Safe Action 4-Layer Integration - 5-Seed Sweep (DEC-0011 v0.3)

> Date: 2026-07-27
> Status: **STRONG NEGATIVE - t = -3.71 (statistically significant)**
> Script: \projects/project_a_self_improvement/code/full_integration_v2.py\
> Command per seed: \--n-ppo-steps 100000 --n-train-episodes 200 --n-eval-episodes 50 --target-fpr 0.10 --safe-action 2 --out-tag v3\
> Sweep: seeds 0-4 (5 seeds, run in parallel: ~17 min total)

## 1. Per-seed v0.3 results (safe_action=2, LunarLander main engine)

| Seed | Ungated | Gated | **Delta** | Val AUROC | Cal threshold | Avg gates |
|------|---------|-------|-----------|-----------|----------------|-----------|
| 0    | 78.3    | -1086.8 | **-1165.1** | 0.963 | 9.3e-03 | 175 |
| 1    | 18.0    | -64.7   | **-82.7**   | 0.803 | 6.1e-01 | 24 |
| 2    | -2.7    | -976.9  | **-974.2**  | 0.994 | 8.9e-02 | 141 |
| 3    | 57.5    | -820.0  | **-877.5**  | 1.000 | 9.0e-09 | 132 |
| 4    | 11.6    | -476.8  | **-488.4**  | 0.717 | 4.5e-01 | 80 |

**All 5 seeds NEGATIVE. 0/5 positive delta. t = -3.71 (significant at p<0.05).**

## 2. Aggregate v0.3 (n=5, sample std)

\\\
Ungated PPO mean:        32.5 +/- 33.9
Gated (Monitor+safe) mean: -685.1 +/- 416.3
Delta (Gated-Ungated):  -717.6 +/- 432.2
Avg gates per episode:  110.4 (range 24 - 175)
Seeds with positive delta: 0 / 5
t-statistic:             -3.71 (df=4, p < 0.05 SIGNIFICANT)
\\\

## 3. Three-way comparison v0.1 vs v0.2 vs v0.3

| Metric | v0.1 (Q-BoN) | v0.2 (calibrated Q-BoN) | v0.3 (safe_action=2) |
|--------|--------------|-------------------------|-----------------------|
| n_eval | 5 | 50 | 50 |
| Ungated | 55.0 +/- 50.3 | 112.5 +/- 52.9 | 32.5 +/- 33.9 |
| Gated | 76.6 +/- 34.0 | -45.6 +/- 230.7 | **-685.1 +/- 416.3** |
| **Delta** | **+21.5 +/- 67.1** | **-158.1 +/- 208.6** | **-717.6 +/- 432.2** |
| t-stat | 0.72 | -1.69 | **-3.71 (significant)** |
| Pos seeds | 3/5 | 0/5 | 0/5 |

**v0.3 is the first result that is statistically significantly NEGATIVE** (t=-3.71, p<0.05).

The progression: each subsequent intervention made the gating loop more aggressive (higher avg_gates, lower calibrated threshold), which amplified the policy destruction.

## 4. Why v0.3 is so bad

### 4.1 The safe-action heuristic is naive

\safe_action=2\ on LunarLander-v3 is "fire main engine" (vertical thrust).
When the Monitor says "high failure prob", the agent just thrusts up.
This destroys trajectories that were already in stable descent or
side-thrust maneuvering.

A more principled safe action would depend on state (e.g., main engine
only if y > 0.5 and vy < 0). But that requires domain knowledge
beyond what the Monitor provides.

### 4.2 The gating trigger frequency is still high

Even with safe_action (not Q), the Monitor still fires on 110/500
steps on average (22%). That's a lot of forced main-engine thrusts
per episode. Even if 50% of those were "correct" (prevented crash),
the other 50% disrupt good trajectories.

### 4.3 The Monitor is detecting pattern, not prescribing action

The SlotMonitor was trained to predict episode-level failure from
slot-attention over the trajectory. It outputs p(failure | recent
slots). But "high p(failure)" does NOT mean "main engine is good".
The Monitor doesn't know what action to take; it just says
"this state looks like past failures."

To convert Monitor output to action, we need a different signal:
either (a) a Q-function that knows the value of each action in
each state, or (b) a direct action predictor (a separate model).
Both need more training data than the 200 episodes we have.

## 5. Decision record (DEC-0011 v0.3 status: REJECTED)

> **DEC-0011 v0.3 (safe action replacement) is REJECTED.** v0.3 made
> Phase 1.5 strictly worse than v0.2: delta -158 -> -718.
>
> The naive safe-action heuristic confirms the diagnosis from v0.2:
> the bottleneck is the **action selection** under gating, not the
> Monitor. With only 200 PPO rollouts in the training set, neither
> Q-learning (v0.2) nor a fixed action (v0.3) can be reliably better
> than the PPO policy itself.
>
> **v0.1 (Q-BoN, n_eval=5) remains the canonical Phase 1.5 result**:
> delta +21.5 +/- 67.1, 3/5 positive, not significant.
>
> **H1 status (final)**: decoupling signal is REAL at the Monitor-
> prediction level (Sections 4.6-4.8, AUROC delta=0.724 across 5 seeds).
> But conversion to policy gain is UNRESOLVED with available techniques
> on LunarLander-v3 + 100K PPO + 200 train episodes.

## 6. DEC-0011 v0.4 candidates (next steps)

A. **Larger Q training set (1000+ PPO rollouts)** — most direct fix
   to the CQL coverage problem. Cost: 5-10 hours of background runs.
B. **Different env** — LunarLander is a hard test (continuous control
   with sparse reward). Try CartPole-v1, Acrobot, MountainCar where
   failure modes are more discrete.
C. **Direct action predictor (imitation)** — train a network to
   predict the best action from the same slot input the Monitor uses.
   This avoids Q-learning entirely.
D. **Halt the gating project** — accept that Monitor + Q-BoN is not
   a useful online intervention on this env, and focus on the
   Monitor-prediction result (which is solid).

## 7. Reproducibility

\\\
Repo commit at run time: f103826 (DEC-0011 v0.2)
Code: full_integration_v2.py (added --safe-action arg)
Launcher: experiments_log/_run_v3_5seed.ps1 (5 parallel)
Per-seed raw logs: experiments_log/_v3_seed{0..4}.log
Aggregator: experiments_log/_agg_v3.py
\\\

## 8. Artifacts

- Code (already tracked in f103826): \code/full_integration_v2.py\
  (with --safe-action flag)
- Per-seed checkpoints (not in git): \checkpoints/full_integration_v3_..._seed{0..4}/phase2_log.json\
- Summary JSON: \experiments_log/phase15_v0p1_v0p2_v0p3_summary.json\
- Per-seed raw logs: \experiments_log/_v3_seed{0..4}.log\ (in .gitignore)
