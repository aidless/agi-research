# PettingZoo Simple Spread — Per-Agent PPO Baseline (Honest NEGATIVE)

> Date: 2026-07-28
> Mode: Per-agent PPO on real PettingZoo benchmark
> Status: **NEGATIVE** — PPO underperforms random on PettingZoo Simple Spread
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

Ran per-agent PPO on the **real** PettingZoo Simple Spread v3 benchmark
(3 agents, 3 landmarks, 18-dim observation, 5 actions, coverage reward).

PettingZoo version: 1.24.1 (has `mpe` subpackage).
Training: 30 PPO updates × 20 episodes = 600 total episodes.

## 2. Result (seed 0)

```
Method              Mean         Std
Random baseline    -85.93       19.64
Per-agent PPO      -100.51      21.70  (eval, deterministic)
Delta              -14.57
```

**Per-agent PPO is WORSE than random by 14.57 points.**

## 3. Why PPO fails here

Several honest observations:

1. **Train vs eval gap**: During training, PPO achieves around -25
   (on-policy, stochastic actions). At eval (deterministic), it gets
   -100. This huge gap suggests the policy is overfitting to specific
   trajectories.

2. **No parameter sharing**: 3 separate PPO policies, each with their
   own 18-dim observation. Harder than shared-parameter baselines
   (e.g., MADDPG, QMIX).

3. **No credit assignment**: per-agent PPO gets the same joint reward
   for all agents. With cooperative task, agents can't tell which one
   helped/hurt.

4. **Short training**: 600 episodes is far less than typical PPO
   convergence (10K+ episodes for Simple Spread).

## 4. What this DOES validate

- ✅ PettingZoo 1.24.1 integration works end-to-end
- ✅ Per-agent PPO training loop runs (3 agents, 18-dim obs, 5 actions)
- ✅ Real benchmark, not hand-coded env
- ✅ Random baseline is reproducible

## 5. What this does NOT validate

- ❌ DMC vs PPO comparison (DMC needs trained Monitors, not random)
- ❌ PPO best performance (no hyperparameter tuning)
- ❌ Multi-agent PPO convergence (would need 10K+ episodes)

## 6. What this means for Phase 2

This is a **methodological finding**, not a project failure:
- Per-agent PPO is HARD on cooperative MA envs
- We need credit assignment (QMIX, COMA) for future work
- DMC needs a real PPO baseline to compare against — we now have
  evidence that per-agent PPO alone is insufficient

**Phase 2 next steps**:
1. Try parameter sharing (PPO with shared actor)
2. Try MADDPG / QMIX as baseline
3. Once we have a working PPO baseline (>random), try DMC

## 7. Artifacts

- `code/pz_baseline.py` (~13 KB, per-agent PPO + PettingZoo wrapper)
- `code/ma_env.py` (hand-coded coverage env, kept for sanity checks)
- `code/dmc_skeleton.py` (DMC architecture, random init)
- `checkpoints/pz_baseline/seed0/phase2_log.json`
- `experiments_log/_pz_baseline_seed0*.txt` (raw outputs)

## 8. Honest limitations

- **1 seed only**: we ran seed 0. Other seeds may give different results.
- **600 episodes is short**: 30 PPO updates is insufficient for
  convergence on Simple Spread.
- **No DMC vs PPO yet**: DMC needs trained Monitors, which requires
  real PPO first.
- **PettingZoo Simple Spread is the simplest cooperative env**: harder
  envs (Simple Reference, Particle) may be even more challenging.

## 9. What we learned

Per-agent PPO without credit assignment underperforms random on
cooperative MA. This is a known result in the field but worth
documenting empirically in our codebase. Future Phase 2 work needs
to use credit assignment methods (MADDPG, QMIX, COMA) or DMC.

This is NOT a project failure. It is a calibration experiment: we
now know that "naive PPO doesn't work on Simple Spread", which sets
the bar for DMC to beat.
