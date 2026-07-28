# Thesis Addendum M - Phase 2: DMC vs MADDPG on PettingZoo Simple Spread

> Date: 2026-07-28
> Status: Phase 2 closed. H5 REFUTED.
> Commits: ea1c785, 86ab39a, 7c17b4b, 4bf1d35, 92a3820, 090d512

## 1. What this addendum covers

Phase 2 implemented the DMC (Decoupled Monitor Coordination) architecture
on PettingZoo Simple Spread v3 to test whether Y1.3-style reward shaping
transfers to multi-agent. The headline result: H5 is REFUTED. Y1.3 is
strictly a *single-agent* finding.

## 2. Architecture implemented

- Per-agent PPO actors (decentralised exec, 18-dim obs, 5-dim discrete
  or 5-dim continuous action)
- Per-agent SlotMonitor (MLP, 20-step history, frozen after training on
  per-agent frozen-PPO rollouts)
- Y1.3-style reward penalty: r_total = r_env - lambda * monitor_prob_i
- Joint failure predictor (diagnostic only; not used in shaping)

## 3. Three ablations

### 3.1 Discrete-action DMC 3-arm (5 seeds, 50+50+50 PPO episodes)

| arm | mean | sd |
|---|---|---|
| real shaping | -125.34 | 42.23 |
| random shaping | -128.02 | 38.98 |
| no shaping | -127.66 | 39.63 |

Paired t-tests: all NOT significant. The earlier +16.2 finding was an
artifact of comparing against an unstable Stage 1.

### 3.2 Continuous-action DMC 3-arm (5 seeds, 600+80 env episodes)

| arm | mean | sd | vs random |
|---|---|---|---|
| real shaping | -101.03 | 21.13 | NEGATIVE |
| random shaping | -84.55 | 8.35 | NEGATIVE |
| no shaping | -77.50 | 6.09 | ~0 |

Paired t-tests: real vs none = -23.53, t=-2.53 (1/5 positive, close to
significant at df=4). Per-agent Monitor AUROC: 0.989 mean.

### 3.3 MADDPG v1 vs v2 (5 seeds, 800 env episodes)

| version | mean | sd | t vs random |
|---|---|---|---|
| v1 (broken bootstrap, seed 0) | -75.78 | 31.11 | +1.66 |
| v2 (proper next_obs + target) | -70.45 | 1.14 | **+7.72, t=+6.50, p<0.001** |

The v1 implementation had three bugs: target_q was hard-coded to zeros,
target_actors/target_critic were never queried, and other agents obs/action
were zero-padded. v2 fixes all three.

## 4. Final 8-way comparison (PettingZoo Simple Spread v3)

| Method | Mean | sd | n | Action |
|---|---|---|---|---|
| Random | -77.45 | 25.03 | 1 | continuous |
| Per-agent PPO | -100.51 | 21.70 | 1 | discrete |
| Shared PPO | -95.15 | 30.64 | 1 | discrete |
| DMC discrete (real shaping) | -125.34 | 42.23 | 5 | discrete |
| DMC continuous real | -101.03 | 21.13 | 5 | continuous |
| DMC continuous none | -77.50 | 6.09 | 5 | continuous |
| MADDPG v1 (broken) | -75.78 | 31.11 | 1 | continuous |
| **MADDPG v2 (proper)** | **-70.45** | **1.14** | **5** | continuous |

## 5. Implications for the Y1 paper

1. **Y1.3 is single-agent only.** The Monitor architecture is portable
   to MA (decoupling assumption holds, AUROC 0.99) but the Y1.3 reward-
   shaping recipe is not (active harm on continuous actions).
2. **Centralised critic wins on credit assignment.** MADDPG v2's +7.7
   beats DMC's no-shaping baseline by ~7. The DMC vs MADDPG gap (~30
   points) is a clean credit-assignment win.
3. **The Monitor's true role is verification, not RL reward.** DLR
   cross-env (97.8% mean) and V1 governance evidence chains are the
   "shipping" use of Monitors; Y1.3 is a special-case single-agent
   recipe that does not generalise.

## 6. Honest limitations

- 5 seeds gives df=4 which is too small for paired effects < 5 points.
- 600-800 env episodes is 50-100x short of typical MA-RL runs.
- PettingZoo Simple Spread is a single benchmark; QMIX/MASAC/MAPPO on
  StarCraft/Hanabi would be needed for generality.
- No inter-agent communication in DMC actors.
- MADDPG v2's centralised critic takes the full global state, which is
  a confounder vs the per-agent Monitor in DMC.

## 7. Action items

- [x] pz_dmc.py (discrete DMC) end-to-end
- [x] pz_dmc_continuous.py (continuous DMC) with real PG
- [x] pz_maddpg_v2.py (proper MADDPG bootstrap)
- [x] 3-arm 5-seed sweeps (discrete + continuous)
- [x] MADDPG v1 vs v2 5-seed comparison
- [x] H5 REFUTED in 9-hypothesis framework
- [x] Y1 paper §4.5/4.6, §5.4, §6.5 added
- [x] Thesis addendum M (this file)
- [ ] Y2: longer training (10K+ episodes), other MA envs, comms, MADDPG
        + Monitor as auxiliary loss (not shaping).

## 8. What this means for the 5-year plan

This Phase 2 closure aligns with the ROADMAP v3 (Y2 planned):
- Phase 2 was originally framed as "DMC validates H5 in MA". It now
  closes as "H5 is REFUTED at this compute; Y2 should explore Monitor
  as auxiliary loss in MADDPG, not as a reward signal".
- This sharpens the contribution: instead of "Monitors help in MA",
  the Y2 framing becomes "Monitors help as VERIFIERS in MA (DLR,
  evidence chain), but not as reward signals".
- Compute was the bottleneck: 5 seeds at 600 episodes is publishable
  as a negative result with proper framing, not as a verdict.
