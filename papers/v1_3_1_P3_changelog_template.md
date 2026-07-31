# Y5 v1.3.1 P3 Result Changelog (TEMPLATE -- fill in after P3 completes)

**Status:** Template (will be filled in when P3 hybrid pre-reg completes)
**Date template:** 2026-08-01 (P3 execution day 1)
**Pre-reg reference:** experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md

## P3 hybrid pre-reg result (TBD)

**Outcome:** [PENDING -- to be filled when P3 completes at ~01:00-01:30]

- If Proposition 3 VALIDATED (hybrid - DLR >= +0.05 with p<0.05 Bonferroni):
  - P3 promoted from Proposition to Theorem in Y5 Section 7.6.2
  - Y5 v1.3.1 → P3 status: VALIDATED (Theorem)
  - Framework implications:
    - Monitor + DLR hybrid is empirically validated (not just a Proposition)
    - P3 hybrid pre-reg framework for future auxiliary signal designs
  - Citation: "P3 (Monitor + DLR hybrid > either alone) empirically validated
    in cooperative MARL (Y3 environment, n=20 paired seeds, p<0.05 Bonferroni)
    -- see pre-reg PROP3-HYBRID for protocol"

- If Proposition 3 REFUTED (hybrid - DLR < +0.05 OR p >= 0.05):
  - P3 retained as Proposition (no change to framework)
  - Y5 v1.3.1 → P3 status: REFUTED (Proposition retained)
  - Framework implications:
    - DLR alone is sufficient; Monitor + DLR hybrid does not add value
    - This is consistent with P4 (cross-task consistency) prediction
  - Citation: "P3 (Monitor + DLR hybrid > either alone) REFUTED in cooperative
    MARL (Y3 environment, n=20 paired seeds, p>=0.05) -- the Proposition is
    retained as an open prediction. The verified shipping use of the
    Monitor remains Pattern A (runtime guardrail) or Pattern B (DLR-in-critic),
    not the hybrid."

- If P3 result is INCONCLUSIVE (high variance, not pre-registered verdict):
  - Document the inconclusive result and defer to a follow-up pre-reg
  - Y5 v1.3.1 → P3 status: INCONCLUSIVE
  - Re-run with larger sample (n=100) per pre-reg Amendment 2 (would need
    a new pre-reg)

## P3 hybrid preliminary results (from 2026-08-01 monitoring)

Note: these are the monitor_only preliminary deltas from the in-progress
run, not the final hybrid verdict.

| Arm | n seeds | Mean delta vs random | Status |
|---|---|---|---|
| monitor_only | 20 / 20 | **+8.4 to +9.2** (POSITIVE) | DONE |
| dlr_only | 7 / 20 | (in progress) | IN PROGRESS |
| v8 (Hybrid) | 0 / 20 | (not yet run) | PENDING |

**Preliminary observation** (2026-08-01 00:42): monitor_only produces a
**+8.4 to +9.2 mean improvement** over the random baseline across
20 paired seeds. This is a POSITIVE signal for the Monitor architecture
in the Y3 cooperative multi-agent setting.

**Implication for P3 verdict**: if v8 (Hybrid) is also +8 or higher,
then v8 - dlr_only < +0.05 (since dlr_only is historically +6 in
similar settings) and P3 is REFUTED. If v8 is +13 or higher, P3
could be VALIDATED.

Full verdict requires all 60 jobs to complete (~01:25 on 2026-08-01).

Note: these deltas are for individual arms vs random baseline, NOT the
hybrid - DLR contrast that the pre-reg verdict is based on. The final
verdict requires the bootstrap contrast computation after all jobs
complete.

## Build artifacts (post P3 completion)

After P3 completes (~01:00-01:30), the following artifacts will be
generated:

- experiments_log/_h10_p3_bootstrap.json (cross-arm contrasts)
- experiments_log/_p3_hybrid_<arm>_s<seed>.log (per-job logs, already exist)
- experiments_log/_p3_hybrid_<TS>.done (completion marker, will be created by launcher)
- papers/y5_monitor_transfer_synthesis.md (will be updated to v1.3.1 with P3 result)
- papers/y5_v1_3_master_synthesis.{tex,pdf,docx,html} (will be regenerated)
- Updated bootstrap analysis in §5.3.1 or new §5.3.3

## Threshold for STOP-PAPER vs EXTEND

Per the pre-reg, the decision rule is:

- **STOP-PAPER-REFUTED-EXTEND**: if F-J contrast (Hybrid - DLR alone) >= +0.05
  with p<0.05 Bonferroni, EXTEND to n=100 paired seeds
- **STOP-PAPER-REFUTED**: if F-J < +0.05 OR p >= 0.05, write paper
  as REFUTED Proposition
- **STOP-PAPER-VALIDATED**: if F-J >= +0.10 with p < 0.01, write
  separate paper on hybrid architecture

The tightened pre-reg (n=20 x 3 arms) is sufficient for the
STOP-PAPER verdict at alpha=0.05 but not for alpha=0.01. The full
pre-reg (n=100 x 3 arms) would be needed for EXTEND or VALIDATED
verdicts.

## Follow-up actions (post P3 verdict)

Regardless of verdict:
- Update Y5 v1.3.1 with P3 result in §5.3.x (or new section)
- Update Y5 Section 7.6.2 (Proposition 3 status)
- Update Y5 Section 7.6.6 (Monotonicity: does P3 observe R1, R2, R3, R4?)
- Update Y5 Section 8.5 Pattern D (Hybrid pre-reg result)
- Re-run reviewer simulator on v1.3.1
- Update COLM 2026 cover letter (if P3 affects abstract or conclusions)
- Update Obsidian Y5 v1.3.1 note

If P3 VALIDATED: promote to NeurIPS 2026 / ICML 2027 workshop paper
If P3 REFUTED: P3 becomes a strong negative result consistent with the
  framework's existing predictions
If P3 INCONCLUSIVE: defer to n=100 pre-reg (would need additional compute)