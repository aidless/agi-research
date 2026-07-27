# Model-Based Planning with slot WM + DLR verifier — STRONG NEGATIVE

> Date: 2026-07-27
> Mode: MBP replaces PPO at inference with planning over DLR safety
> Status: **NEGATIVE** — same fundamental issue as DLR verifier gating
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we tried

After DLR verifier gating failed (replacing PPO with do-nothing degrades
performance), we tried model-based planning: use the slot world model to
predict next state for each candidate action, then evaluate DLR safety,
pick the action with max safety.

```
For each action a in {0, 1, 2, 3}:
  1. Predict next_slot via slot WM: slot_t + a -> slot_{t+1}
  2. Decode slot to predicted_obs
  3. Evaluate DLR predicates on predicted_obs -> safety score
Pick action with max safety score.
```

## 2. Pipeline

`projects/project_a_self_improvement/code/mbp_slot_dlr.py`:
- SlotAttention + per-slot dynamics + linear decoder
- Trained on 12K (obs, action, obs_next) triples from PPO
- 20 epochs, MSE 0.0017 (much higher than slot_dynamics.py 0.000007
  because reconstruction includes decoder error)

## 3. Result (seed 0, 50% MBP + 50% PPO mixing)

| metric | value |
|--------|-------|
| Ungated PPO mean | 114.5 +/- 124.7 |
| MBP-gated mean | -158.6 +/- 134.9 |
| Delta | **-273.10** |

## 4. Why MBP fails

1. **WM reconstruction error**: predicted_obs has WM error, DLR predicates
   evaluated on noisy predictions produce noisy safety scores.
2. **Mixing** (50% MBP, 50% PPO) means we override PPO half the time with
   worse predictions, hurting overall performance.
3. **The fundamental issue**: replacing PPO actions with anything (do-nothing,
   Q-BoN, behavior-clone, MBP) hurts on LunarLander because PPO is already
   strong.

## 5. What this means

This is the FOURTH inference-time intervention to fail:
- v0.1-v0.4C: Monitor gating (6 experiments)
- DLR verifier gating (P1.3)
- MBP (P1.4)

**Pattern**: inference-time intervention does not work on LunarLander with
current techniques. The right answer is **training-time regularization**
(Y1.3, +50 over baseline), not inference-time replacement.

## 6. Y1 direction (refined again)

After 4 inference-time failures, the Y1 direction is now:

1. **Y1.3 (training-time regularizer)** — already POSITIVE (+50)
2. **Y1.4 (Monitor as PPO value baseline)** — use Monitor as variance
   reduction signal during PPO training.
3. **Y1.5 (Model-based policy improvement)** — use world model + DLR to
   generate synthetic training data, then train PPO on augmented data.

The pattern is clear: auxiliary signals are valuable as **training
constraints**, not as **interventions**.

## 7. Artifacts

- `code/mbp_slot_dlr.py` (~300 lines)
- `code/checkpoints/mbp_slot_dlr/seed0/phase2_log.json`
- `experiments_log/_mbp_slot_dlr_seed0.txt` (raw output)
- Compute: ~2.5 min per run on CPU
