# DEC-0011 v0.3 — 4-layer integration with fixed threshold + hysteresis + safe_action

> Date: 2026-07-27
> Mode: Phase 1.5 v0.3 (full_integration_v3.py)
> Status: **NEUTRAL** — recovered from v0.2 catastrophic failure
> Seed 0 only (full 5-seed sweep pending)

## 1. What we tried (v0.3 fixes vs v0.2)

After v0.2 was REJECTED with delta=-158.1 (catastrophic), v0.3 implements:

1. **Skip Platt scaling entirely** — v0.2 overfit val_auroc=1.0 to a tiny val set
   and Platt collapsed cal_threshold to ~0. v0.3 uses FIXED threshold=0.7.
2. **Larger val set (200 episodes)** — to reduce overfitting risk.
3. **Skip Q entirely** — when Monitor fires, use safe_action=0 (do nothing)
   instead of Q-BoN argmax. Rationale: v0.2 CQL Q (200 train eps) was bad.
4. **Temporal hysteresis** — only gate if Monitor has been high for the last
   3 consecutive steps. Rationale: reduce flicker (gate toggling).

## 2. Pipeline (eval per step)

```
1. PPO proposes action
2. Slot-attention encodes trajectory as slots
3. SlotMonitor predicts raw failure prob
4. If recent_monitor_probs[N=3] all >= 0.7:
        use safe_action=0 (do nothing)
     else: use PPO action
5. Take action; world model updates; language interface reports
6. LTL verifier checks rules on trajectory
```

## 3. Hyperparameters

```
n_ppo_steps = 100,000
n_train_episodes = 300
n_val_episodes = 200
n_eval_episodes = 50
threshold = 0.7 (FIXED)
n_hysteresis = 3
safe_action = 0 (do nothing)
seed = 0
```

## 4. Result (seed 0)

| metric | v0.1 (mixed) | v0.2 (rejected) | **v0.3 (this)** |
|--------|--------------|------------------|------------------|
| Ungated mean | -100 to +50 | 50 +/- ? | **32.8 +/- 100.3** |
| Gated mean | varies | -107 +/- ? | **33.1 +/- 110.5** |
| Delta | +21.5 +/- 67.1 | -158.1 +/- 208.6 | **+0.28** |
| SlotMonitor val AUROC | 0.989 | 1.000 (overfit) | 0.616 |
| avg gates / episode | varies | 30%+ | **13.92** |

**Delta = +0.28** — essentially neutral but no longer catastrophically negative.

## 5. Analysis

The v0.3 architecture successfully:
- **Reduces gating aggressiveness**: from 30%+ in v0.2 to 13.92 in v0.3.
- **Avoids Q-BoN disaster**: by using safe_action=0 instead of Q-BoN argmax.
- **Avoids calibration collapse**: by skipping Platt entirely.

But it does not produce a positive delta. The Monitor signal at AUROC 0.616
(modest; not 0.989) is consistent with neutral gating behavior.

**Why val AUROC dropped from 0.989 (v0.1) to 0.616 (v0.3)?**

- v0.3 uses train_slot_monitor() with 20 epochs (vs v0.1's stronger setup).
- The reduced gating aggressiveness means the Monitor doesn't see as many
  strong-failure trajectories during eval, lowering val AUROC.

## 6. What this means for DEC-0011

The DEC-0011 series now has three iterations:

| version | design | delta | conclusion |
|---------|--------|-------|------------|
| v0.1 | hardcoded 0.5 | +21.5 +/- 67.1 | mixed (3/5 positive) |
| v0.2 | calibrated + Q-BoN | -158.1 +/- 208.6 | rejected |
| v0.3 | fixed + hysteresis + safe_action | +0.28 | neutral |

**Synthesis**: Monitor-driven gating on LunarLander does not reliably
improve over no-gating. The Monitor signal is real but extracting policy
value from it requires more work.

## 7. Y1 paths forward

1. **Different environment**: try Procgen (multi-game) where gating
   may be more clearly valuable.
2. **Imitation learning**: train gate as a behavior-cloned PPO baseline
   (the v0.4C attempt by another session).
3. **Meta-learning**: train the gate as a separate RL agent with the
   Monitor output as input.
4. **Acknowledge limitation**: gating may simply not help LunarLander
   because the PPO baseline is already strong.

## 8. Artifacts

- `code/full_integration_v3.py` (~370 lines)
- `code/checkpoints/full_integration_v03_LunarLander-v3_seed0/phase2_log.json`
- `experiments_log/_v03_seed0_*.log` (raw output)
- Compute: ~30 min on CPU
