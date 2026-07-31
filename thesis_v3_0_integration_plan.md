# Thesis v3.0 Integration Plan (2026-08-01)

**Status:** Plan (no actual thesis content yet)
**Source thesis:** thesis_draft_v2.0.tex (159 KB, 158918 bytes, 3199 lines)
**Target thesis:** thesis_draft_v3.0.tex (~250 KB expected)
**Coordinator:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Driver:** Y5 master synthesis v1.3 + Y1/Y3/Y4 companion papers all at v1.0

## Why v3.0

The thesis v2.0 (2026-07-30) was written before:
- Y5 v1.3 master synthesis existed (current v1.3, 89 pages, §7.6 framework)
- Y1/Y3 papers were upgraded to v1.0 with Y5 cross-references
- Y4 v0.6.1 GSM8K 200-token follow-up was completed
- 6 cross-task meta-analytic methods were applied (Fisher / Stouffer / Bonferroni / Holm / Hedges g / forest plot)
- 7 references added to bibliography (Shimodaira / Cover & Thomas / Valiant / Haussler / Hanley-McNeil / Holm / Hedges)
- §7.6.6 formal monotonicity proved (16-cell Boolean lattice)
- §7.6.3 cost-weighted observation table (4-Refutation GPU budget)
- Pre-Reg for Proposition 3 hybrid test (2026-08-01 to 2026-08-15 window)

The thesis v3.0 integrates all of these into a coherent thesis structure.

## What changes from v2.0 to v3.0

### 1. New abstract / executive summary (~5 pages)

Add a new high-level abstract summarizing the 11-comparison cross-context
record and the §7.6 formal framework. This becomes the thesis's lead
content and replaces the v2.0 lead.

### 2. Update Part VII (Cross-Environment & Transfer)

v2.0 Part VII discusses cross-environment transfer (H3 DLR predicate transfer).
v3.0 expands this to include the cross-context synthesis (RL -> MARL -> LLM)
and the 11-comparison record.

**New subsection:**
- \section{Cross-context synthesis: 11 empirical comparisons}
  - Y1 single-agent RL (1 VALIDATED)
  - Y3 multi-agent MARL (5/6 REFUTED + 1 partial)
  - Y4 LLM self-monitoring (4/4 REFUTED)
  - Combined-p meta-analysis (6 methods converge on H10 REFUTATION)
  - Forest plot visualization

### 3. Add new Part X (Y5 Formal Framework)

After Part IX (Project G / Y4 LLM), add a new Part X titled
"Cross-context Synthesis: The Failure-Prediction Monitor Does Not Transfer"
that contains the Y5 master synthesis content.

**New subsections:**
- \section{The 11-comparison cross-context record}
- \section{Y5 Section 7.6: formal framework}
- \section{Y5 Section 7.5.5: first-principles motivation}
- \section{Y5 Section 7.6.2: Assumption A1}
- \section{Y5 Section 7.6.6: formal monotonicity proof}
- \section{Y5 Section 7.6.3: cost-weighted observation}
- \section{Y5 Section 8.5: deployment patterns}
- \section{Y5 Section 9.6: framework limitations}
- \section{Y5 Section 5.3.1 + 5.3.2: cross-task meta-analysis}

### 4. Update Part IX (Project G / Y4 LLM)

Update the Y4 chapter to v1.0 status with Y5 cross-references.

**Changes:**
- Frontmatter: v1.0 status with Y5 cross-reference
- New subsection: "Y5 Connection: How Y4 fits in the 11-comparison record"
- References to §5.3.1 + §5.3.2 combined-p meta-analysis
- References to §7.6.3 Refutation R2 (LLM Monitor without retraining)

### 5. Update Part II (Project A: Self-Improvement via Decoupled Monitors)

Update the Y1 chapter to v1.0 status with Y5 cross-references.

**Changes:**
- Frontmatter: v1.0 status
- New subsection: "Y5 Connection: How Y1 fits in the 11-comparison record"
- References to §7.6 framework (Y1 as the 1 VALIDATED case)

### 6. Update Part VI (Project F: Multi-Agent)

Update the Y3 chapter to v1.0 status with Y5 cross-references.

**Changes:**
- Frontmatter: v1.0 status with AAMAS 2027 target venue
- New subsection: "Y5 Connection: How Y3 fits in the 11-comparison record"
- References to §7.6 Condition 1 violation
- References to Proposition 3 hybrid test (Pre-Reg committed, execution 2026-08-01)

### 7. Update bibliography (References section)

Add 7 new references:
- Shimodaira 2000 (covariate shift)
- Cover & Thomas 1991 (information theory)
- Valiant 1984 (PAC-learning)
- Haussler 1990 (PAC sample complexity)
- Hanley-McNeil 1982 (AUROC SE)
- Holm 1979 (step-down correction)
- Hedges 1981 (bias-corrected d)

### 8. Update addenda

Add new addendum for 2026-08-01:
- Y5 v1.3 master synthesis paper submitted to COLM 2026
- Y1, Y3, Y4 all at v1.0 status with Y5 cross-references
- Proposition 3 hybrid test scheduled for 2026-08-01 to 2026-08-15 window
- OpenReview submission package built (arxiv_submission.tar.gz)
- Pre-registration chain: 4 documents (original + Amendment 1 + Addendum + Prop3-Hybrid)

## Estimated work

| Task | Effort | Blocking? |
|---|---|---|
| New abstract / executive summary | 0.5 day | No |
| Update Part VII | 1 day | No |
| New Part X (Y5 framework) | 2 days | No |
| Update Part IX (Y4) | 0.5 day | No |
| Update Part II (Y1) | 0.5 day | No |
| Update Part VI (Y3) | 0.5 day | No |
| Update bibliography | 0.25 day | No |
| New addendum | 0.25 day | No |
| LaTeX rebuild + PDF render | 0.5 day | Yes |
| **Total** | **6 person-days** | |

## Sequencing

- [ ] Day 1: Update Parts II, VI, IX (3 days of work, all in parallel since each is independent)
- [ ] Day 2: Write new Part X content from Y5 source material
- [ ] Day 3: Update bibliography, add addendum, rebuild LaTeX
- [ ] Day 4: Render PDF, run reviewer simulator on the new thesis sections

## Notes

The thesis v3.0 is a research-direction document, not a separate
submission. The COLM 2026 submission is Y5 v1.3 alone. The thesis v3.0
provides the broader context for the Archimedes Project across all 5
sub-projects (Y1 / Y3 / Y4 / Y5 / P3 hybrid).

A simplified v3.0 (just adding Y5 cross-references to the existing thesis
structure, no new Part X) could be done in 1 day and is a reasonable
intermediate step.

The full v3.0 (with new Part X) is best done after the Y5 paper has been
accepted at COLM 2026, so the thesis can cite the published Y5 paper
rather than the preprint.

## Current status (2026-08-01)

- Y5 v1.3 master synthesis: COMPLETE (89 pages, COLM 2026 submission)
- Y1 v1.0 paper upgrade: COMPLETE
- Y3 v1.0 paper upgrade: COMPLETE
- Y4 v1.0 paper upgrade: COMPLETE
- OpenReview submission package: BUILT
- Proposition 3 hybrid test: IN PROGRESS (background launch, expected complete 2026-08-01 evening)
- Thesis v3.0: PLAN (this document)
- Y5 v1.3.1 (P3 result): PENDING (after P3 completes)

## Files needed for v3.0 build

Source materials:
- papers/y5_v1_3_master_synthesis.tex (need to convert .md to .tex)
- papers/y1_paper_draft.md (already in markdown, needs .tex conversion)
- papers/monitor_signal_vs_dlr_6pathway.md (already in markdown)
- papers/project_g_v0_5_h10_paper.md (already in markdown)
- papers/cover_letter_colm2026_v1_3.md
- papers/reviewer_simulator_output_v1_3.md
- papers/supplementary_S16_version_history.md

Build artifacts:
- thesis_draft_v3.0.tex (target)
- thesis_draft_v3.0.pdf (target)
- thesis_draft_v3.0.md (target, for Obsidian)

The v3.0 build can start once the Y5 markdown source has been
converted to LaTeX (separate task).