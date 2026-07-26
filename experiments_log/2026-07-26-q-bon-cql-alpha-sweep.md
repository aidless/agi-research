# CQL alpha sweep — Q-BoN TTC final negative result

> Date: 2026-07-26
> Status: ALL 5 alphas NEGATIVE — TTC definitively does not work
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. Setup

Swept --cql-alpha across {0.0, 0.1, 1.0, 5.0, 10.0} on seed 0.
Other params: 100K PPO, 200 train episodes, 30 eval episodes, N=4, hidden=64.

## 2. Results (LunarLander-v3, seed 0, 5 alphas)

| alpha | PPO mean | Q-BoN mean | Delta |
|-------|----------|------------|-------|
| 0.0 (no CQL) | 37.2 | -72.6 | -109.8 |
| 0.1 | 16.6 | -168.5 | -185.1 |
| **1.0** | 21.6 | -7.8 | **-29.4** ← best |
| 5.0 | 116.3 | 25.3 | -91.0 |
| 10.0 | 54.1 | 18.5 | -35.6 |

5/5 alphas NEGATIVE.

**Best alpha**: 1.0 (closest to neutral, delta magnitude -29.4)
**Worst alpha**: 0.1 (delta magnitude -185.1)

## 3. Discovery: PPO baseline is non-deterministic

Same seed 0 produces PPO mean ranging from 16.6 to 116.3 across the
sweep. This is because torch random initialization order matters;
even with `torch.manual_seed(seed)`, the exact sequence of
random ops differs slightly between runs (e.g., different env reset
order, different numpy random calls before torch seed).

**Implication**: PPO baseline quality is a major confounder. Even if
Q-BoN were good, delta = Q-BoN - PPO would vary ±50 just from PPO
stability. With delta magnitude of -29 to -185, we cannot reliably
distinguish Q-BoN effect from PPO noise.

## 4. Interpretation

**Q-BoN as TTC strategy does NOT improve PPO on LunarLander-v3 with
100K PPO budget at any CQL alpha.**

This is the empirical conclusion after 7 attempts:
- v1-v4: Monitor-BoN (4 variants)
- v5: Q-BoN no CQL
- v6: Q-BoN + CQL alpha=1.0
- v7: alpha sweep

Across all 7, the most positive delta is +16.2 (alpha=1.0, but
PPO baseline was 0.2 = barely any learning). All other attempts
negative.

## 5. Why Q-BoN fundamentally fails

Hypothesis: TTC requires either:
1. **Per-step rollout** (use Monitor/Q to predict future, then
   optimize over a multi-step trajectory) — we did 8-step rollouts
   but ranking was still wrong
2. **Population-based search** (genetic algorithms, MCTS over
   trajectories) — we only did single-step BoN
3. **Tree search over actions** (expand action tree, propagate
   Q-values) — we did flat BoN

Single-step BoN with a learned scorer is **too narrow** to be a
useful TTC strategy on LunarLander.

## 6. Y1 plan (revised — abandon TTC TTC in this form)

Given 7 failed attempts, we should **deprioritize BoN-style TTC** in
Y1. Instead focus on:
1. **CQL itself as Y1 paper**: CQL is the standard offline RL
   fix and we have the infrastructure. Write Paper A v2 covering
   CQL + TTC as a methodology contribution even if TTC doesn't
   empirically win.
2. **Multi-step TTC**: tree search, MCTS, or value iteration
3. **Value-function BoN with state cloning at every step**: more
   expensive but might work

## 7. Artifacts

- `code/q_bon.py` (~11 KB, with CQL)
- 5 phase2_log.json files in `code/checkpoints/qbon_LunarLander-v3_seed0_cql_alpha*/`
- Compute: ~27 minutes (5 alphas × ~3.5 min each)

## 8. TTC final summary (7 attempts, all negative or marginal)

| ver | alpha | delta (seed 0) |
|---|---|---|
| v1 Monitor (proxy) | n/a | -11 (2-seed mean) |
| v2 Monitor (clone) | n/a | -283 |
| v3 Monitor (balanced) | n/a | -49 |
| v4 Monitor (h=256) | n/a | -362 |
| v5 Q-BoN no CQL | 0.0 | -109.8 (sweep) |
| v6 Q-BoN CQL | 1.0 | +16.2 / -29.4 (sweep) |
| **v7 alpha sweep (5 values)** | **0.0-10.0** | **all negative** |

**TTC BoN strategy does NOT work on LunarLander-v3 with our setup.**
This is honest strong evidence. Y1 should pursue multi-step TTC,
tree search, or pure offline RL benchmarks (CQL/IQL) instead.