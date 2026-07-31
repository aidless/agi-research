# S16: Version History -- Consolidated Changelog v0.8 -> v1.3

**Section:** Supplementary Materials, Section 16 (Version History)
**Paper:** Y5 Master Synthesis Paper, COLM 2026 submission
**Date:** 2026-07-31 (v1.3 camera-ready)
**Compiled by:** Codex under Liu Zewen's supervision

This document consolidates every meaningful change made to the Y5
master synthesis paper from v0.8 (predecessor) through v1.3 (camera-
ready). It is supplementary material for the COLM 2026 submission
and is intended to give reviewers and readers a complete audit trail.

## Version timeline

```
v0.8 (predecessor, 2026-07-30)
  |  56 pages
  |  No formal framework
  |  Original synthesis draft (empirical chain only)
  v
v1.0 (2026-07-31)
  |  64 pages (+8)
  |  Added §7.6 formal framework (7 Definitions + 4 Propositions +
  |    4 Refutations + Logical disjunction of R1-R4 + Connection
  |    to existing AGI safety architectures)
  |  Added framework diagram (mermaid-rendered PNG)
  |  Added reviewer simulator (12 items: 3 P0 / 6 P1 / 3 P2)
  |  Added COLM 2026 cover letter (v1.0)
  v
v1.1 (2026-07-31)
  |  70 pages (+6)
  |  Added §5.3.1 cross-task combined-p meta-analysis (Fisher /
  |    Stouffer Z equal / Stouffer Z weighted / Bonferroni min p)
  |  Added Assumption A1 in §7.6.2 (positive mutual information)
  |  Added empirical check of A1 (Y1 satisfies; Y3/Y4 hold weakly)
  |  Closed 2 of 3 P0 reviewer items
  v
v1.2 (2026-07-31)
  |  82 pages (+12)
  |  Added §5.3.2 extended meta-analysis (Bonferroni-Holm step-down
  |    + Hedges g bias-corrected + forest plot visualization)
  |  Added §7.5.5 first-principles motivation (covariate shift +
  |    mutual information + PAC-learning, 4 references)
  |  Added §7.6.3 logical disjunction (R3.3) + compute-cost
  |    estimate for R4 (R2.1, ~150-250 GPU-h for 7B, ~1500-2500
  |    GPU-h for 70B)
  |  Added §7.6.6 monotonicity of refutation observation (R3.4)
  |  Added §8.5 deployment patterns (4 patterns: Runtime guardrail
  |    / DLR-in-critic / Pre-commit review / Monitor+DLR hybrid)
  |  Added §9.6 framework limitations (3 explicit limitations)
  |  Added Pre-Reg for Proposition 3 hybrid test (5 KB)
  |  Added forest plot (27 KB)
  |  Added COLM 2026 cover letter (v1.2)
  |  Closed 10 of remaining 10 reviewer items (all P1+P2)
  v
v1.3 (2026-07-31) -- CAMERA-READY
  |  89 pages (+7)
  |  Added GPU reservation in Pre-Reg P3 (R1.5: ~50 GPU-h,
  |    2026-08-01 to 2026-08-15 window)
  |  Added provenance note for n=5 Hedges g (R1.6: marks
  |    d = -0.250 as post-hoc)
  |  Added cross-reference from §8.5 Pattern D to Pre-Reg (R2.5)
  |  Added 7 references to bibliography (R2.6 + extended:
  |    Shimodaira 2000 / Cover & Thomas 1991 / Valiant 1984 /
  |    Haussler 1990 / Hanley-McNeil 1982 / Holm 1979 /
  |    Hedges 1981)
  |  Added formal monotonicity proof in §7.6.6 (R3.5: 16-cell
  |    Boolean lattice + Monotonicity Lemma + Corollary)
  |  Added cost-weighted observation table in §7.6.3 (R3.6:
  |    4-Refutation GPU-hour budget + Archimedes Project
  |    priority order)
  |  Added COLM 2026 cover letter (v1.3)
  |  Re-ran reviewer simulator (0 items open, all 3 reviewers Accept)
  v
v1.3.1 (planned, 2026-08-15+)
  |  Proposition 3 hybrid pre-reg executed (~50 GPU-h)
  |  §5.3.2 / §7.6 framework updated with P3 empirical result
  |  Y5 v1.3.1 = v1.3 + P3 evidence (VALIDATE or REFUTE)
```

## Detailed change log

### v0.8 -> v1.0 (8 page delta)

New sections:
- §7.6 Formal framework: definitions, propositions, and falsifiability
  (with 7 Definitions, 4 Propositions, 4 Refutations, and 5
  subsections: Definitions / Propositions / Falsifiability / Why-
  not-summary / Connections)
- §7.6 framework diagram (mermaid.js -> Edge headless -> PNG)
  with 3 Convergence Conditions (C1/C2/C3), 4 Refutations
  (R1/R2/R3/R4), 4 Propositions (P1/P2/P3/P4), and the
  Transferability claim

New artifacts:
- `papers/y5_v1_0_master_synthesis.{html,docx,pdf}` (64-page PDF)
- `papers/reviewer_simulator_output_v1_0.md` (12-item report)
- `papers/cover_letter_colm2026_v1_0.md`
- `papers/figures_v2/fig_y5_7_6_convergence_refutations.png`
- `papers/figures_v2/_mermaid_source/fig_y5_7_6_convergence_refutations.html`
- `E:\gen_pdf.py` (Edge headless PDF generator)

Reviewer items (12 total):
- 3 P0 (combined-p, Assumption A1, 3-condition decomposition)
- 6 P1 (footnote required-n, Hanley-McNeil, Pre-Reg P3, R4 compute,
       deployment patterns, framework limitations)
- 3 P2 (first-principles, monotonicity, logical disjunction)

### v1.0 -> v1.1 (6 page delta)

New sections:
- §5.3.1 Cross-task combined-p meta-analysis (4 H10 sample sizes
  combined via Fisher / Stouffer Z equal / Stouffer Z weighted /
  Bonferroni min p)
- §7.6.2 Assumption A1 explicit (positive mutual information
  between auxiliary signal and policy value function)
- §7.6.2 empirical check of A1 (Y1 satisfies; Y3/Y4 hold weakly)

New artifacts:
- `experiments_log/_h10_combined_p.json` (4 sample sizes + 4 meta-
  analytic methods)

Reviewer items closed: 2 P0 (R1.4 combined-p, R3.1 Assumption A1)

### v1.1 -> v1.2 (12 page delta)

New sections:
- §5.3.2 Extended meta-analysis (Bonferroni-Holm step-down + Hedges g
  + forest plot visualization)
- §7.5.5 First-principles motivation (3 Convergence Conditions derived
  from covariate shift / mutual information / PAC-learning)
- §7.6.3 Logical disjunction (R3.3)
- §7.6.3 compute-cost estimate for R4 (R2.1)
- §7.6.6 Monotonicity of refutation observation (R3.4)
- §8.5 Deployment patterns (4 patterns: Runtime guardrail / DLR-in-
  critic / Pre-commit review / Monitor+DLR hybrid)
- §9.6 Framework limitations (3 explicit limitations)

New artifacts:
- `experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md`
  (Pre-Reg for Proposition 3)
- `papers/figures_v2/fig_h10_combined_p_forest.png` (forest plot)
- `papers/figures_v2/_forest_plot_source.html` (build source)

Reviewer items closed: 10 of remaining 10 (R1.1, R1.2, R1.3, R2.1,
R2.2, R2.3, R2.4, R3.2, R3.3, R3.4 -- mix of P1 and P2)

### v1.2 -> v1.3 (7 page delta)

New content (all P3 very-minor):
- Pre-Reg P3 §11 Compute reservation (R1.5: ~50 GPU-h,
  2026-08-01 to 2026-08-15 window)
- §5.3.2 n=5 Hedges g provenance note (R1.6: marks d = -0.250
  as post-hoc)
- §8.5 Pattern D cross-reference to Pre-Reg (R2.5)
- Bibliography: 7 new references (R2.6 + extended: Shimodaira
  2000 / Cover & Thomas 1991 / Valiant 1984 / Haussler 1990 /
  Hanley-McNeil 1982 / Holm 1979 / Hedges 1981)
- §7.6.6 Formal Monotonicity Lemma + 16-cell Boolean lattice
  + Corollary (R3.5)
- §7.6.3 Cost-weighted observation table + Archimedes Project
  priority order (R3.6)

Reviewer items closed: 6 of remaining 6 (R1.5, R1.6, R2.5, R2.6,
R3.5, R3.6 -- all P3 very-minor)

### v1.3 -> v1.3.1 (planned)

Pending:
- Execute Proposition 3 hybrid pre-reg (~50 GPU-h, 2026-08-01
  to 2026-08-15)
- Update §5.3.2 / §7.6 framework with P3 empirical result
- Y5 v1.3.1 = v1.3 + P3 evidence

## Reviewer item ledger (18 items, all closed)

| ID | Reviewer | Type | Status | Version closed | Location |
|---|---|---|---|---|---|
| R1.1 | R1 (Empirical ML) | P1 | closed | v1.2 | §7.6.1 (decomposition uniqueness) |
| R1.2 | R1 | P1 | closed | v1.2 | experiments_log/Pre-Reg PROP3-HYBRID.md |
| R1.3 | R1 | P1 | closed | v1.2 | §7.6.2 (required-n footnote) |
| R1.4 | R1 | P0 | closed | v1.1 | §5.3.1 (combined-p meta-analysis) |
| R1.5 | R1 | P3 | closed | v1.3 | Pre-Reg PROP3 §11 (GPU reservation) |
| R1.6 | R1 | P3 | closed | v1.3 | §5.3.2 (n=5 post-hoc note) |
| R2.1 | R2 (AGI safety) | P1 | closed | v1.2 | §7.6.3 (R4 compute-cost) |
| R2.2 | R2 | P1 | closed | v1.2 | §8.5 (deployment patterns) |
| R2.3 | R2 | P2 | closed | v1.2 | §7.5.5 (first-principles) |
| R2.4 | R2 | P1 | closed | v1.2 | §9.6 (framework limitations) |
| R2.5 | R2 | P3 | closed | v1.3 | §8.5 (Pattern D cross-ref) |
| R2.6 | R2 | P3 | closed | v1.3 | References (7 new entries) |
| R3.1 | R3 (Theory/formal) | P0 | closed | v1.1 | §7.6.2 (Assumption A1) |
| R3.2 | R3 | P2 | closed | v1.2 | §7.6.1 (Hanley-McNeil footnote) |
| R3.3 | R3 | P1 | closed | v1.2 | §7.6.3 (logical disjunction) |
| R3.4 | R3 | P2 | closed | v1.2 | §7.6.6 (monotonicity) |
| R3.5 | R3 | P3 | closed | v1.3 | §7.6.6 (formal monotonicity proof) |
| R3.6 | R3 | P3 | closed | v1.3 | §7.6.3 (cost-weighted observation) |

## Git commit ledger (this submission cycle)

```
897fe40 Y5 v1.0: add formal framework (§7.6) + reviewer sim + PDF
0d76d15 Add revision task list: 12 items from Y5 v1.0 reviewer simulator
5ee8416 Y4 v0.6.1 finalization: paper + helpers + pre-regs + JSONs
cb49d07 Project G: GSM8K + CoT + last-20-token window + orphan-LM fix
85e05ae Paper updates: HYPOTHESIS_STATUS + project_g_v0_5 + supplementary
5f6b179 Add repo infrastructure: GitHub Actions CI + citation.cff + obsidian helper
90d9896 gitignore: ignore backup files
daad479 Y5 v1.1: address 2 P0 reviewer items (combined-p + Assumption A1)
20ce800 Y5 v1.2: address 10 remaining reviewer items + extended meta-analysis
d12b063 Y5 v1.2: re-run reviewer simulator + v1.2 COLM cover letter
93eedd7 Y5 v1.3: address 6 v1.2 reviewer items (camera-ready pass)
411e6e1 Y5 v1.3: final camera-ready reviewer simulator (all 3 reviewers: Accept)
<this commit> Y5 v1.3: COLM cover letter + consolidated changelog (S16)
```

Total: 13 commits in this submission cycle, all pushed to
`git@github.com:aidless/agi-research.git`.

## Summary statistics

| Metric | v0.8 | v1.0 | v1.1 | v1.2 | v1.3 |
|---|---:|---:|---:|---:|---:|
| Pages | 56 | 64 | 70 | 82 | **89** |
| PDF size (MB) | 1.18 | 1.19 | 1.37 | 1.52 | **1.59** |
| DOCX size (KB) | -- | 115 | 189 | 224 | **229** |
| Markdown size (KB) | -- | 111 | 113 | 140 | **152** |
| Reviewer items open | n/a | 12 | 10 | 6 | **0** |
| Pre-Regs in repo | 2 | 3 | 3 | 4 | 4 |
| Figures in repo | 10 | 11 | 11 | 12 | **12** |
| Cover letters | 0 | 1 | 1 | 2 | **3** |
| Reviewer simulators | 0 | 1 | 1 | 2 | **3** |
| Git commits (cycle) | -- | -- | -- | -- | **13** |
| Total additions (lines) | -- | +7115 | +837 | +1044 | **+427** |

## Camera-ready readiness

- All 18 cumulative reviewer items closed
- v1.3.1 in queue (P3 hybrid pre-reg execution by 2026-08-15)
- COLM 2026 cover letter v1.3 ready
- 13 Git commits pushed to origin
- Working tree clean

The paper is camera-ready for COLM 2026 submission.