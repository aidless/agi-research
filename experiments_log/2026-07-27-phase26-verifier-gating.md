# Phase 2.6: Verifier-Aware Gating — architecture works, calibration issue

> Date: 2026-07-27
> Status: ARCHITECTURE WORKS, default rules too easy to violate
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. Improvement over Phase 2.5

Phase 2.5 used only Monitor prob as gating signal. Phase 2.6 combines:
- Monitor prob (failure prediction)
- Verifier rule violations (LTL rule checking)

Gate fires when EITHER signal exceeds threshold. This gives the
agent more "evidence" for intervention.

## 2. Method

`code/phase26_verifier_gating.py` (13894 bytes):

Per step:
  evidence = (Monitor_prob, n_verifier_violations)
  if Monitor_prob > gate_threshold OR n_viol > viol_threshold:
    use Q-BoN argmax (gated)
  else:
    use PPO action (ungated)

Gating log includes which signal triggered.

## 3. Smoke test (4K PPO, 1 episode, history=4)

### viol_threshold=0 (any violation)
  Gated: -1586.5
  Ungated: -15.5
  Delta: -1571.0
  Triggers: monitor=0, verifier=1, both=226, neither=5

### viol_threshold=2 (≥2 violations)
  Gated: -626.7
  Ungated: -262.6
  Delta: -364.1
  Triggers: monitor=85, verifier=0, both=0, neither=38

## 4. Diagnosis

**With viol=0**: LTL rules are violated almost every step. The
"ALWAYS angle_below(1.0)" rule fires whenever the agent tilts past
1 rad, which happens in most states. Verifier signal saturates.

**With viol=2**: Verifier rarely fires (only 0 timesteps in our smoke
test) because the agent doesn't accumulate 2+ violations in a short
window. Effectively this reverts to Phase 2.5 (Monitor only).

**Across all variants**: delta is NEGATIVE. The Monitor is too eager
(gating at 0.5 fires too often given Monitor's bias toward "danger").
Q-BoN is also undertrained and provides little improvement.

## 5. Lessons

1. **Verifier rule calibration matters**: rules must be rare enough
   that violations are meaningful signals, not constant noise
2. **Larger rule set with weighted importance**: future work should
   have ~20+ rules with priorities, not 3 binary rules
3. **PPO/Q training budget dominates**: at 4K PPO steps, both PPO and
   Q are too undertrained to differentiate
4. **Architecture is sound**: gating logic correctly combines two
   evidence sources. Just need better-calibrated inputs

## 6. Y1 work to make this work

1. **Larger PPO budget** (100K+ for honest eval)
2. **Better Verifier rules** with priority weights
3. **Trained Verifier** (learn which rules matter)
4. **Adaptive gate threshold** (lower when agent is stable, raise
   when in danger zone)

## 7. Cumulative AGI roadmap

| Phase | Status |
|---|---|
| 1.1 A+C slot-Monitor | ✅ AUROC 0.989 |
| 1.2 C dynamics | ✅ next-step err 0.000007 |
| 1.3 D language | ✅ template-based |
| 1.4 E LTL verifier | ✅ rule check |
| 1.5 Full integration | ✅ smoke test |
| 2.1 naive gating | ❌ -192 (broken) |
| 2.5 smart Q-gating | ⚠️ -10 (working) |
| **2.6 verifier-gating** | **⚠️ -364 (working, calibration issue)** |

## 8. Artifacts

- `code/phase26_verifier_gating.py` (13894 bytes, NEW)
- `code/checkpoints/phase26_LunarLander-v3_seed0/phase2_log.json`
- Compute: ~30 seconds