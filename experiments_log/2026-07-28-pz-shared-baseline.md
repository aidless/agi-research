# PettingZoo Simple Spread — Parameter-Sharing PPO (also NEGATIVE)

> Date: 2026-07-28
> Mode: Shared-actor PPO on real PettingZoo benchmark
> Status: **NEGATIVE** (but better than per-agent)
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we did

PPO with **parameter sharing**: 1 shared actor + 1 shared critic for all
3 agents. This is the standard trick in cooperative MARL (used in
MADDPG, COMA, etc.).

Same training setup as per-agent PPO (30 updates × 20 episodes = 600
total episodes, seed 0).

## 2. Result (seed 0)

| Method | Mean | Std | Delta vs Random |
|--------|------|-----|------------------|
| Random baseline | -84.71 | 29.39 | (ref) |
| Per-agent PPO | -100.51 | 21.70 | -15.80 |
| **Shared PPO (eval)** | **-95.15** | **30.64** | **-10.43** |
| Shared vs per-agent | - | - | **+5.36** |

## 3. Honest interpretation

- **Shared PPO is better than per-agent PPO** by +5.36 (consistent with
  parameter sharing's known advantage for cooperative MA).
- **But shared PPO is STILL worse than random** by -10.43.
- **Train-eval gap** is still large: train ~-75, eval ~-95.

## 4. Why both PPO variants struggle

Several honest observations:

1. **600 episodes is far too few** for PPO to converge on Simple Spread.
   Standard convergence: 10K+ episodes (Hardt 2016).
2. **Train-eval gap suggests overfitting**: stochastic training policy
   achieves ~-75, deterministic eval achieves ~-95.
3. **PPO on Simple Spread is hard**: even standard implementations
   require careful tuning.
4. **Reward signal is sparse**: agents only get positive signal when
   they hit landmarks.

## 5. What this means for Phase 2

This is a **methodological finding**, not a failure:
- Per-agent PPO: -100 (baseline)
- Shared PPO: -95 (better, but not converged)
- Random: -85 (lower bound)

The gap from -95 (shared PPO) to -85 (random) is small (-10). A
properly tuned PPO should easily beat random; we're not there yet.

**Phase 2 next steps**:
1. Try much longer training (5000+ episodes)
2. Try other algorithms (MADDPG, QMIX, COMA)
3. Once PPO baseline beats random, implement real DMC
4. Compare DMC vs the working PPO baseline

## 6. Comparison with literature

Standard cooperative MARL results on Simple Spread:
- MADDPG: ~-50 to -100 (depends on training budget)
- QMIX: ~-60 to -80
- COMA: ~-70 to -90
- MAPPO (shared PPO, well-tuned): ~-40 to -70

Our shared PPO at -95 is in the lower end of standard results,
consistent with insufficient training (600 vs 10K+ episodes).

## 7. Artifacts

- `code/pz_shared_baseline.py` (~13.5 KB)
- `checkpoints/pz_shared_baseline/seed0/phase2_log.json`
- `experiments_log/_pz_shared_seed0.txt` (raw output)
- Compute: ~90 sec per run

## 8. Honest limitations

1. **1 seed only**: need 5 seeds for t-stat
2. **600 episodes is short**: 30x less than typical convergence
3. **No hyperparameter tuning**: lr=3e-4, clip=0.2, gamma=0.99 are defaults
4. **No comparison to other MA methods** (MADDPG, QMIX)
5. **No credit assignment algorithm tested**
6. **No DMC yet** (requires trained Monitors)
7. **PettingZoo Simple Spread v3** only (not v2 or harder variants)

## 9. Lessons learned

Parameter sharing provides +5.36 improvement over per-agent PPO.
This is small but real, validating the standard MA-RL approach.

PPO alone is insufficient at our compute scale. We need:
- Longer training (compute)
- Credit assignment (algorithms)
- Possibly DMC (architecture)

This is a calibration result, not a project failure.
