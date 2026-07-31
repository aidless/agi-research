# Y5 v1.3 -> v1.3.1 transition plan (2026-08-01, P3 hybrid pre-reg day 1)

**Status:** Plan (will execute after P3 hybrid pre-reg completes)
**Driver:** P3 hybrid verdict (VALIDATE / REFUTE / INCONCLUSIVE)
**Date scheduled:** 2026-08-01 ~01:30 (after P3 completes)
**Executor:** Codex under Liu Zewen supervision

## Pre-conditions for v1.3.1 build

1. P3 hybrid pre-reg completes (~01:30)
2. Bootstrap aggregator runs (`python _agg_p3_hybrid.py`)
3. Verdict recorded in `experiments_log/_p3_hybrid_bootstrap.json`

## v1.3.1 changes by verdict

### If P3 VALIDATED (Hybrid - DLR >= +0.05 with p<0.05)

- Y5 Section 7.6.2 Proposition 3 status: VALIDATED (promoted to Theorem)
- Y5 Section 7.6.3 Refutation 1 (R1): NOT OBSERVED (still open)
- Y5 Section 5.3.1 cross-task meta-analysis: add P3 row
- Y5 Section 7.5.5 first-principles: confirmed for hybrid case
- Y5 Section 7.6.6 monotonicity: confirmed (P3 result strengthens framework)
- Y5 Section 8.5 Pattern D: status changed from "proposed" to "validated"
- Y5 Section 7.5.5 first-principles: extended to cover the Hybrid case
- Y5 Section 9.6 framework limitations: remove the "P3 untested" limitation
- Y5 v1.3.1 title: "with the P3 hybrid validation" (or similar)
- COLM 2026 cover letter: add P3 result to the contribution list
- Reviewer simulator v1.3.1: re-run, expected 0-2 items
- OpenReview package: rebuild with v1.3.1

### If P3 REFUTED (Hybrid - DLR < +0.05 OR p >= 0.05)

- Y5 Section 7.6.2 Proposition 3 status: REFUTED (retained as Proposition
  with empirical evidence against it)
- Y5 Section 7.6.3 Refutation: NOT OBSERVED
- Y5 Section 5.3.1 cross-task meta-analysis: add P3 row (consistent with
  the cross-task REFUTATION pattern)
- Y5 Section 7.6.6 monotonicity: confirmed
- Y5 Section 8.5 Pattern D: status changed from "proposed" to
  "proposed but REFUTED in n=20 v1.3.1"
- Y5 Section 7.5.5 first-principles: extended to discuss why the
  hybrid is REFUTED (likely the Monitor + DLR combination has correlated
  failure modes that the framework's Monotonicity Lemma predicts)
- Y5 Section 9.6 framework limitations: add "P3 REFUTED in n=20 v1.3.1"
  as a new limitation
- Y5 v1.3.1 title: "with the P3 hybrid REFUTATION" (or similar)
- COLM 2026 cover letter: add P3 result to the negative findings list
- Reviewer simulator v1.3.1: re-run, expected 0-2 items
- OpenReview package: rebuild with v1.3.1

### If P3 INCONCLUSIVE (high variance, not pre-reg verdict)

- Y5 v1.3.1: defer to v1.3.2 (after running full n=100 pre-reg)
- Y5 v1.3.1 status: "P3 hybrid pre-reg inconclusive; n=100 follow-up scheduled"
- v1.3.1 builds with P3 status as "PENDING FULL PRE-REG"
- COLM 2026 submission: still v1.3 (no v1.3.1 needed for COLM 2026)
- Reviewer simulator: not re-run for v1.3.1 (use v1.3 result)
- OpenReview package: stay at v1.3

## Build artifacts (all verdicts)

- `papers/y5_v1_3_1_master_synthesis.{md,html,docx,pdf}` (NEW v1.3.1 artifacts)
- `papers/y5_v1_3_master_synthesis.{md,html,docx,pdf}` (KEPT as v1.3 archive)
- `papers/v1_3_1_P3_changelog.md` (renamed from template, filled in)
- `experiments_log/_p3_hybrid_bootstrap.json` (UPDATED with final data)
- `experiments_log/_p3_hybrid_<arm>_s<seed>.log` (60 files, all done)
- `papers/cover_letter_colm2026_v1_3_1.md` (NEW, supersedes v1.3)
- `papers/reviewer_simulator_output_v1_3_1.md` (NEW)
- Updated Obsidian `Y5 v1.3.1 Master Synthesis` note

## v1.3.1 build pipeline (4 steps)

1. Update Y5 markdown source with P3 result section
2. Render Y5 v1.3.1 .html, .docx, .pdf via pandoc + Edge headless
3. Update cover letter, reviewer sim, Obsidian note
4. Commit + push to origin

## Estimated work

- 30-60 min of focused text work
- ~5 min of git/pandoc/Edge build
- Total: ~1 hour from P3 completion to v1.3.1 ready

## Timeline

- 00:05: P3 launch
- 00:35: monitor_only done
- 00:43: dlr_only partial (preliminary verdict: P3 likely REFUTED)
- ~01:00: dlr_only done
- ~01:30: all 60 done; v1.3.1 build starts
- ~02:30: v1.3.1 ready, committed, pushed