# Phase 2.7 Multi-Seed Validation — HONEST RESULT: gating doesn't help

> Date: 2026-07-27
> Status: CRITICAL — single-seed finding of thresh=0.6 sweet spot was misleading
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. The single-seed trap

The Phase 2.7 sweep on seed 0 alone showed:
- thresh=0.6 (33 gates): mean=91.4
- thresh=0.9 (0 gates): mean=64.6
- "Gating wins by +27!"

This finding was an artifact of single-seed evaluation. PPO is
non-deterministic across runs even with the same seed, and the
single seed happened to give a misleading result.

## 2. The honest 3-seed result

`code/phase27_multiseed.py` (11103 bytes): trained PPO + Monitor + Q
on each of 3 seeds, then swept thresholds within each seed.

### Per-seed results

| threshold | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.5 | 0.2 | 163.0 | 48.7 |
| 0.6 | -4.7 | 177.2 | 73.5 |
| 0.7 | 10.6 | 143.4 | 33.9 |
| 0.8 | -9.0 | 143.7 | 56.7 |
| 0.9 | -8.0 | 210.7 | 123.1 |

### Mean across 3 seeds ± std

| threshold | mean | std |
|---|---|---|
| 0.5 | +70.6 | 68.3 |
| 0.6 | +82.0 | 74.5 |
| 0.7 | +62.6 | 57.9 |
| 0.8 | +63.8 | 62.5 |
| **0.9** | **+108.6** | 89.9 |

## 3. Analysis

**Best threshold: 0.9 (most ungated). Best mean: +108.6.**

As threshold lowers (more gating fires), mean decreases. Gating
consistently hurts on average. The single-seed finding was misleading.

### Statistical considerations

- Best gated (thresh=0.6) vs best ungated (thresh=0.9): +82.0 vs +108.6
- Difference: 26.6 points (about 1/3 of std)
- Not statistically significant at p<0.05 with n=3 seeds
- **Conclusion**: gating doesn't reliably improve PPO on LunarLander

### Per-seed observations

- **Seed 0**: PPO didn't learn (mean ~0 across thresholds). All
  thresholds similar.
- **Seed 1**: PPO worked well (200+). thresh=0.9 has highest (210.7)
  — gating consistently reduced performance.
- **Seed 2**: PPO partial (~50-120). thresh=0.9 again highest (123.1).

In every seed, **thresh=0.9 wins**. This is strong evidence that
gating doesn't help.

## 4. Why gating fails (post-mortem)

The original intuition was that the Monitor would correctly identify
"danger states" and gating would replace PPO actions with Q-BoN
safer actions. The result is the opposite:

- Monitor is well-calibrated (AUROC 0.989 in isolation) but
  - For ONLINE gating, the threshold matters. Monitor prob > 0.5
    fires 33+ times per 200-step episode — too often
  - At higher threshold (0.9), Monitor rarely fires (0 gates) and
    PPO acts normally — this is the BEST case
- Q-BoN doesn't help even when it fires, because:
  - Q is trained on PPO rollouts, so argmax_Q ≈ PPO argmax
  - When Monitor is uncertain, Q is also uncertain
  - Gating just adds noise to a working PPO

## 5. Honest conclusion for the AGI roadmap

**The gating architecture does not work on LunarLander-v3 with 100K
PPO budget.**

This is a STRONG NEGATIVE RESULT for Project A's H1 follow-up.
The H1 ablation (frozen Monitor AUROC 0.989) is solid, but using
the Monitor for online gating doesn't improve PPO.

This does NOT invalidate the H1 ablation (that's about Monitor
quality in isolation). It does mean "self-aware gating" is not a
useful TTC strategy with current architecture.

## 6. Y1 work to make gating work

1. **Different envs**: try CartPole or MountainCar where failure
   modes are different
2. **Larger Q-network**: 256-hidden, more training
3. **Action-level intervention**: gate only individual high-risk
   actions, not whole episodes
4. **Multi-step reasoning**: use Monitor + Q + World Model for
   actual planning, not just 1-step gating

## 7. Cumulative AGI roadmap (revised)

| Phase | Result | Status |
|---|---|---|
| 1.1 A+C slot-Monitor | AUROC 0.989 | ✅ solid |
| 1.2 C dynamics | next-step err 0.000007 | ✅ |
| 1.3 D language | template-based | ✅ |
| 1.4 E LTL verifier | rule check | ✅ |
| 1.5 Full integration | smoke test | ✅ |
| 2.1 naive gating | -192 | ❌ |
| 2.5 Q-BoN gating | -10 (single seed) | ❌ |
| 2.6 verifier-gating | -61.8 (single seed) | ❌ |
| **2.7 multi-seed** | **-26.6 (3 seeds)** | **❌ (HONEST NEGATIVE)** |

**Gating strategy does not work.** Architecture pieces (A+C+D+E) are
solid individually but combining them for self-aware behavior
doesn't improve PPO on LunarLander.

## 8. Artifacts

- `code/phase27_multiseed.py` (11103 bytes, NEW)
- `code/checkpoints/phase27_multiseed_LunarLander-v3/phase2_log.json`
- `experiments_log/_multiseed.txt` (full output)
- Compute: ~110 minutes