# Pre-Registration: Proposition 3 Hybrid Test (Monitor + DLR > Either Alone)

**Author:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-07-31
**Status:** DRAFT (v1.2 addition to Y5 master synthesis)
**Companion:** Proposition 3 in `papers/y5_monitor_transfer_synthesis.md` Section 7.6.2
**Reviewer item addressed:** R1.2 (P1) in `.tasks/task-20260731-y5v10-revisions.md`

## 1. Hypothesis

**H_p3 (Proposition 3 prediction).** In the cooperative multi-agent setting of the Y3 paper, an architecture that combines (a) the Monitor as a runtime verification signal AND (b) DLR predicates as a per-step shaping bonus will produce a larger positive effect on the policy reward than either (a) or (b) alone.

**H_null.** The Monitor + DLR hybrid produces an effect that is statistically indistinguishable from the larger of (Monitor alone effect, DLR alone effect).

## 2. Decision rule

The hypothesis is **VALIDATED** if:
- The hybrid (Monitor + DLR) effect on the policy reward is +0.05 or larger than the DLR-alone effect at n=100 paired seeds
- AND the difference has p < 0.05 (two-sided, Bonferroni-corrected for 3 arm-comparisons: hybrid vs Monitor, hybrid vs DLR, Monitor vs DLR)

The hypothesis is **REFUTED** if:
- The hybrid effect is NOT +0.05 or larger than the DLR-alone effect
- OR the difference is not statistically significant

**STOP-PAPER criterion:** if the hybrid effect is +0.10 or larger with p < 0.01, write a separate paper on the hybrid architecture. If the hybrid effect is between +0.05 and +0.10 with p < 0.05, integrate the result into the Y5 master synthesis as v1.3 update. If the hybrid effect is below +0.05 or not significant, retain Proposition 3 as an open prediction.

## 3. Architecture

The hybrid architecture combines:

- **Monitor (Pattern A)**: as a runtime verification signal. Each rollout is evaluated by the Monitor; rollouts with predicted failure probability > 0.5 are flagged. Flagged rollouts do NOT contribute to the policy update (they are filtered out from the training batch).

- **DLR predicates (Pattern B)**: as a per-step shaping bonus. The Y3 v8 dlr_only architecture is the baseline; the predicate is "agent should not enter a cell already occupied by a teammate" (or analogous coordination predicate for the test environment).

- **Hybrid**: Monitor filtering + DLR shaping bonus simultaneously. The Monitor filters out high-failure-probability rollouts; the DLR predicate provides the shaping bonus on the remaining rollouts.

The 3-arm comparison is:
1. Monitor alone (Pattern A only): the Y3 v6 architecture
2. DLR alone (Pattern B only): the Y3 v8 dlr_only architecture (which gave +0.06 at n=100)
3. Hybrid (Monitor + DLR): the new architecture

## 4. Environment

Reuse the Y3 cooperative multi-agent environment (6-pathway systematic investigation). Specifically, the v8 dlr_only test environment with the same hyperparameters, seeds, and rollout length as the Y3 paper.

## 5. Sample size

Required n = 100 paired seeds (per Y3 v8 dlr_only standard). This matches the Y3 n=100 protocol. At each seed, run all 3 arms (Monitor alone, DLR alone, hybrid) with the same random initialization.

## 6. Pre-registered analyses

1. Per-arm mean and 95% CI for each of the 3 arms.
2. Hybrid - DLR contrast (the primary test).
3. Hybrid - Monitor contrast (secondary).
4. DLR - Monitor contrast (replicating the Y3 v8 dlr_only finding).
5. Combined-p across the 3 arm-comparisons if any individual arm is borderline.

## 7. Pre-registration timestamp

Written BEFORE data collection on 2026-07-31. No data has been collected at this pre-registration date. If the test is run, the data collection date and any deviations from this pre-registration will be documented in an Amendment.

## 8. Compute budget

Estimate: ~100 seeds x 3 arms x ~10 minutes per seed = ~50 hours wall-clock on CPU. This is within the existing Archimedes Project compute budget and could be run by 2026-08-15 if approved.

## 9. Expected outcome

Based on the framework's P3 prediction (hybrid > either alone) and the Y3 v8 dlr_only baseline of +0.06 at n=100, the expected effect size is +0.08 to +0.12 for the hybrid. This would give 80% power at n=100 for a two-sided test at alpha=0.05/3 = 0.0167. If the true effect is in this range, the test should validate the hypothesis.

If the test REFUTES P3, the framework needs a revision: the hybrid does not provide additional value over DLR alone. This would update Proposition 3 to "Monitor + DLR hybrid is not better than DLR alone" -- a specific, falsifiable claim that can be retained or overturned by future tests.

## 10. Cancellation criterion

The pre-registration is cancelled if:
- The Monitor + DLR combination is found to be infeasible (e.g., the Monitor's runtime overhead exceeds the rollout budget for the multi-agent environment)
- The Y3 v8 dlr_only baseline is itself overturned by a prior replication (would force a re-evaluation of the baseline)

In either case, a new pre-registration would be written before any new data is collected.