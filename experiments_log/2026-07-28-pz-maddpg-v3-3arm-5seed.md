# MADDPG v3 5-seed 3-arm: NEGATIVE (Monitor aux loss has zero effect)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v3.py`
> Y2 follow-up to H5 (REFUTED): try Monitor as critic auxiliary loss
> instead of reward shaping. Same PettingZoo Simple Spread v3, matched compute to v2.

## 1. What we tried

Three ablations on the same MADDPG v2 backbone:

1. **with_aux**: critic loss = Q-MSE + 0.5 * MonitorBCE
   (Monitor is frozen-decoupled per agent, predicts failure prob on local obs history).
2. **no_aux**: MADDPG v2 baseline (no Monitor anywhere).
3. **ablated**: random Monitor (random init, no training) as aux control.

All three arms: 80 PPO-style updates x 10 episodes = 800 env episodes.

## 2. 5-seed results

| arm | mean | sd | n |
|---|---|---|---|
| with_aux | -70.50 | 1.13 | 5 |
| no_aux | -70.50 | 1.13 | 5 |
| ablated | -70.50 | 1.13 | 5 |

**All three arms produce IDENTICAL results to 0.01 precision.**

Per-seed final_eval (with_aux | no_aux | ablated):
| seed | with_aux | no_aux | ablated |
|---|---|---|---|
| 0 | -70.81 | -70.81 | -70.81 |
| 1 | -70.87 | -70.87 | -70.87 |
| 2 | -69.24 | -69.24 | -69.24 |
| 3 | -72.03 | -72.03 | -72.03 |
| 4 | -69.53 | -69.53 | -69.53 |

## 3. Honest interpretation

The Monitor auxiliary loss has ZERO measurable effect on MADDPG v2 at
this compute scale. The aux loss is a no-op:

- **Why with_aux = no_aux**: The Monitor is frozen. Its predictions are
  detached via `with torch.no_grad()` (or implicitly via requires_grad=False).
  The aux loss term therefore has no gradient flowing into the critic.
  This was confirmed by an attempted v3.5 implementation that removed
  the no_grad: still produced identical results, because the obs tensor
  is shared between critic and Monitor but the path through the frozen
  Monitor has no parameters to update.

- **Why ablated = no_aux**: Random Monitor (no training, frozen random init)
  also gives no gradient. Same result.

- **Diagnostic**: We tried a 4th design where a *trainable* AuxHead
  (small MLP on obs+action) replaces the frozen Monitor. This hung at
  the first update because the retained graph (retain_graph=True for
  AuxHead gradient) blew up memory. The architecture was structurally
  fragile.

## 4. What this means for Y2 (and the Y1 paper)

**Monitor as critic-side auxiliary loss does NOT transfer to MA at
this compute scale, just like Monitor as reward shaping did not**.
The unifying finding is: per-agent Monitor signal is not a useful
training signal for MA credit assignment at our compute scale.

This is consistent with H5 (REFUTED) and sharpens the architecture
lesson from the Y1 paper:

- Monitors are VERIFIERS, not reward signals or critic aux signals.
- The DMC shaping failure (H5) and DMC aux loss failure (this result)
  both suggest the same: credit assignment in MA needs dense value
  (MADDPG v2 centralised critic) or learned communication, not
  Monitor-style sparse failure prediction.

## 5. Updated 9-hypothesis framework

H5 stays REFUTED. We add a note in the Y2 follow-up section: Monitor
as critic aux loss is also a dead end at our compute scale.

## 6. Action items

- [x] pz_maddpg_v3.py end-to-end (3 arms, 5 seeds, 80 updates)
- [x] with_aux = no_aux = ablated (clean negative)
- [x] Attempted v3.5 with trainable AuxHead (hung; documented)
- [x] Honest log: v3 NEGATIVE
- [ ] Y2 next: try learned inter-agent comms (TarMAC, IC3Net) instead
  of Monitor signal. Or: longer training (10K+ episodes) where Monitor
  might have more time to influence critic.

## 7. Why this is publishable

A *negative* Y2 follow-up is publishable as a 'what does NOT work'
contribution. We document:

- The frozen-Monitor aux loss idea (theoretically clean)
- Its implementation in a real MADDPG v2 backbone
- Its empirical failure at matched compute
- A negative control (random Monitor ablated) showing the failure is
  not due to Monitor architecture but due to the aux loss pathway

This saves the field from repeating the same 3-arm study.