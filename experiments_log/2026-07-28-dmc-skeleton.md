# Phase 2 DMC SKELETON — Architecture Test (No Training)

> Date: 2026-07-28
> Mode: Phase 2 base implementation
> Status: **SKELETON** — architecture runs end-to-end, no training
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we built

A minimal hand-coded multi-agent env + DMC (Decentralized Monitor
Coordination) architecture skeleton. Two files:

- `projects/project_f_multi_agent/code/ma_env.py` (~5.7 KB)
  - 3-agent coverage env on 5x5 grid with 3 landmarks
  - Per-agent action: 5 discrete (up/down/left/right/no-op)
  - Joint reward: -mean distance to nearest unclaimed landmark
  - Episode ends after 20 steps OR all landmarks covered
- `projects/project_f_multi_agent/code/dmc_skeleton.py` (~6.4 KB)
  - Per-agent SimplePolicy (random init, untrained)
  - Per-agent SimpleMonitor (random init, untrained)
  - JointFailurePredictor (random init, untrained)
  - End-to-end forward pass with shared predicate broadcast

## 2. Result (seed 0, 20 episodes)

| Method | Mean return | Std |
|--------|-------------|-----|
| Random baseline | **-42.40** | 14.29 |
| DMC skeleton (untrained) | -50.74 | 14.87 |

**DMC skeleton is slightly worse than random** because:
- Random init MLPs may sample worse actions than uniform random
- Joint failure predictor is random
- Monitors are random (adds noise to reward shaping)

This is expected and **honest**: untrained policies should not beat random.

## 3. What this DOES validate

- API integration: env + policy + monitor + joint failure predictor
  all connect correctly
- Action space: 5 actions per agent
- Observation space: 8-dim per agent
- Predicate broadcast: each agent's monitor output → joint failure
- Termination: claimed-all or max-steps

## 4. What this does NOT validate

- **No training**: policies are random init, never trained
- **No Monitor learning**: monitors never see real failure data
- **No actual DMC coordination**: the joint failure predictor is random
- **No comparison to baseline**: random vs DMC-skeleton is the only comparison

## 5. Y2 next steps (real implementation)

To get a real DMC result, we need:
1. **PPO per agent** (replace SimplePolicy with trained PPO)
2. **Monitor training per agent** (collect rollouts, train frozen Monitor)
3. **Joint failure predictor training** (use joint failure labels)
4. **Reward shaping** (combine per-agent + joint failure signals)

This is roughly 4 weeks of work for Y2 (2027-01 to 2027-02).

## 6. Honest unknowns

- **Whether DMC actually helps**: decoupling may not transfer to
  multi-agent credit assignment
- **Whether DLR broadcast helps**: symbolic knowledge transfer
  requires per-agent predicate learning + broadcast infrastructure
- **Whether 3 agents is enough**: scaling to 10+ agents may reveal
  broadcast bottlenecks

## 7. Why hand-coded env instead of PettingZoo

- PettingZoo not in our Python env (path issues)
- Building minimal env is faster than fixing package management
- Honest: this is a SKELETON, not a real benchmark
- Future: replace with PettingZoo Simple Spread when Python env fixed

## 8. Artifacts

- `code/ma_env.py` (5.7 KB) — 3-agent coverage env
- `code/dmc_skeleton.py` (6.4 KB) — DMC architecture skeleton
- `checkpoints/dmc_skeleton/seed0/phase2_log.json`
- `experiments_log/_dmc_skeleton_seed0.txt` (raw output)
- Compute: ~10 sec per smoke test
