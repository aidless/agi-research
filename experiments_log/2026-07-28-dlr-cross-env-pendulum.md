# DLR Cross-Env — Pendulum-v1 (4th env)

> Date: 2026-07-28
> Mode: DLR predicates cross-environment test (4th env)
> Status: Consistent with 3-env result; 3-seed mean 98.8%
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Added Pendulum-v1 (3-dim state, continuous action) to the cross-env DLR
test. 3 predicates:
- low_angle (|cos(theta)| near 1): pole near upright
- low_ang_vel (|theta_dot| < 1.0): slow rotation
- upright_approx (cos(theta) > 0.9): very near upright

## 2. Result (3 seeds)

| Predicate | seed 0 | seed 1 | seed 2 | 3-seed mean |
|-----------|--------|--------|--------|-------------|
| low_angle | 0.989 | 0.968 | 0.970 | 0.976 |
| low_ang_vel | 0.996 | 0.975 | 0.997 | 0.989 |
| upright_approx | 0.999 | 0.996 | 0.998 | 0.998 |
| **mean** | **0.995** | **0.980** | **0.988** | **0.988** |

## 3. 4-env DLR cross-env summary

| Env | State | Actions | Predicates | 3-seed mean |
|-----|-------|---------|------------|-------------|
| LunarLander-v3 | 8 | 4 | 7 | 95.5% |
| CartPole-v1 | 4 | 2 | 4 | 98.1% |
| Acrobot-v1 | 6 | 3 | 5 | 98.9% |
| Pendulum-v1 | 3 | 1 (discretized) | 3 | **98.8%** |
| **4-env mean** | - | - | **19** | **97.8%** |

## 4. Honest boundary (per user feedback)

This is consistent with prior DLR results, BUT:
- **All predicates are hand-coded**, not learned. The DLR is fitting a
  simple supervised problem, not discovering structure.
- **Train and test sets are from the same distribution** (random policy
  rollouts). OOD generalization (e.g., does `upright` on LunarLander
  transfer to CartPole''s `upright`?) is **untested**.
- **30 train episodes is small**. Real-world deployment would need more.
- **No peer review**. All numbers are self-validated.

The 4-env consistency is suggestive but not definitive. We should frame
the claim as: "DLR architecture fits hand-coded predicates with ~98%
accuracy across 4 classical-control envs; further generalization
experiments needed."

## 5. Artifacts

- `code/dlr_cross_env.py` (now with Pendulum support)
- `checkpoints/dlr_cross_env/Pendulum-v1_seed{0,1,2}/phase2_log.json`
- `experiments_log/_dlr_pendulum_seed{0,1,2}.txt`
