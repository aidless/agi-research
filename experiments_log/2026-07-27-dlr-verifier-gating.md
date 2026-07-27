# DLR Verifier-Aware Gating — STRONG NEGATIVE on LunarLander

> Date: 2026-07-27
> Mode: First end-to-end DLR pipeline (DLR predicates -> safety formula -> gate)
> Status: **NEGATIVE** — same fundamental issue as DEC-0011
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we tried

After DLR-attention fix (95.5% mean accuracy on 7 predicates), we built
the first end-to-end DLR gating pipeline:

1. Train DLR predicates (95.5% accuracy)
2. Build composite safety formula over predicates
3. At each step, evaluate safety truth value
4. If safety < threshold: substitute safe_action (do nothing)
5. Compare to ungated PPO and LTL-gated baselines

## 2. Pipeline

`projects/project_a_self_improvement/code/dlr_verifier_gating.py`:
- Loads DLR predicates from `dlr_attention.py` (95.5% accuracy)
- Builds safety formula: 0.5 * upright + 0.5 * low_velocity
- Gates when safety < threshold (tried 0.3, 0.4, 0.5)
- Safe action: 0 (do nothing)

## 3. Result (seed 0, threshold sweep)

| threshold | ungated | DLR-gated | delta | avg gates |
|-----------|---------|-----------|-------|-----------|
| 0.5       | 50.1    | -134.7    | -184.8 | 69.6  |
| 0.4       | 82.2    | -130.1    | -212.3 | 56.8  |
| 0.3       | 38.5    | -84.3     | -122.8 | 34.1  |

All three thresholds produce strong negative deltas.

## 4. Why DLR gating fails (and why LTL gating also fails)

The DLR predicates work (95.5% accuracy), but the **gating mechanism** is
wrong: replacing PPO actions with do-nothing (action=0) on LunarLander
*prevents the lander from maneuvering*.

PPO has learned an aggressive policy that violates the upright and
low_velocity predicates during normal flight. Replacing those actions
with "do nothing" causes the lander to drift and crash.

**Safety ≠ do-nothing.** This is the same fundamental issue as DEC-0011
v0.1-v0.4: the Monitor/DLR signal is real, but the intervention
(do-nothing) is wrong for LunarLander.

## 5. What this means for DLR

- The DLR predicates are **valuable for verification** (95.5% accuracy
  on rules like "landed -> in_pad").
- The DLR predicates are **not directly useful for action gating** on
  LunarLander, because the right action when "in danger" is not
  do-nothing — it's *some specific maneuver*.
- The DLR pipeline should be used for **post-hoc verification** or
  **as a critic for policy learning**, not as a real-time gate.

## 6. Y1 direction (refined)

Instead of action-gating, the next direction is:
1. **Model-based planning** (P1.4): use DLR predicates to evaluate
   predicted states, and find the action that maximizes predicted safety.
2. **Policy gradient with DLR baseline**: use DLR as a value baseline
   for variance reduction.
3. **Imitation from safe rollouts**: identify safe PPO rollouts and
   behavior-clone them.

## 7. Artifacts

- `code/dlr_verifier_gating.py` (~250 lines)
- `code/checkpoints/dlr_verifier_gating/seed0/phase2_log.json`
- `experiments_log/_dlr_verifier_gating_seed0*.txt` (3 attempts)
- Compute: ~2 min per run on CPU
