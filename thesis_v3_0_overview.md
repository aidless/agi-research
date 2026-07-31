# Thesis v3.0 -- The Archimedes Project: A Cross-Context Investigation

**Authors:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-08-01 (v3.0)
**Status:** v3.0 .tex source committed; v3.0 PDF build pending v2.0 LaTeX repair
**Supervisor:** Independent researcher
**Keywords:** reinforcement learning, multi-agent systems, LLM self-monitoring, pre-registered empirical methodology, formal framework

---

## Abstract

The Archimedes Project investigates whether the failure-prediction Monitor
training signal transfers across agent contexts. Across 11 independent
pre-registered empirical comparisons in three contexts (single-agent RL,
multi-agent MARL, LLM self-monitoring), the Monitor produces a positive
training-time effect in exactly 1 cell, and that single positive result is
from hand-crafted DLR predicates, not the learned Monitor.

The thesis integrates four papers at v1.0/v1.3:

- **Y1 v1.0** Single-agent RL (LunarLander-v3, n=15, VALIDATED +39.5)
- **Y3 v1.0** Multi-agent MARL (6 pathways, 5/6 REFUTED + 1 partial v8 dlr_only)
- **Y4 v1.0** LLM H10 self-monitoring (4 sample sizes + 2 task families, 4/4 REFUTED)
- **Y5 v1.3** Cross-context synthesis with 6-method meta-analysis and the
  formal Section 7.6 framework (3 Convergence Conditions, 4 Propositions,
  4 named Refutations, formal Monotonicity Lemma)

Key empirical contribution: a 6-method cross-task meta-analysis (Fisher
combined-p / Stouffer Z / Bonferroni min / Bonferroni-Holm / Hedges g /
forest plot) all agree the H10 LLM self-monitoring REFUTATION is robust
across 4 sample sizes and 2 task families (Fisher combined-p = 0.7947,
NOT significant).

Key theoretical contribution: a predictive framework with 4 named
falsifiers. The framework is falsified iff any of R1-R4 is observed.
NONE has been observed across 11 empirical comparisons, supporting the
framework predictive validity claim.

Key practical contribution: 4 concrete deployment patterns (Runtime
guardrail / DLR-in-critic / Pre-commit review / Monitor+DLR hybrid) that
match the verified shipping use of the Monitor as verification, not as
training signal.

Open Research: the Proposition 3 hybrid test is pre-registered with GPU
reservation 2026-08-01 to 2026-08-15. After execution, the result will be
incorporated as v1.3.1 (P3 VALIDATE -> Proposition -> Theorem) or v1.3.1
(P3 REFUTE -> Proposition retained, framework unchanged).

---

## Thesis structure

The thesis has 10 Parts (I-IX from v2.0 + new Part X for Y5):

### Part I: Foundations
- The Monitor architecture (frozen reference + decoupled training)
- The convergence-conditions framework (informal precursor to Y5 Section 7.6)
- PPO + LunarLander-v3 baseline setup

### Part II: Project A -- Self-Improvement via Decoupled Monitors
- Y1 paper (v1.0): single-agent RL, LunarLander-v3
- n=15 seed validation: +39.5 mean improvement over PPO baseline
- Cross-environment DLR predicate transfer (H3)
- **Y5 connection**: Y1 is the 1 VALIDATED case in the 11-comparison
  record; all 3 Convergence Conditions hold simultaneously in Y1

### Part III: Project C -- Causal World Models with Slot Attention
- Slot attention Monitor architecture (H4)
- 0.989 vs 0.796 AUROC on LunarLander-v3 (slot vs raw history)
- Not central to v3.0 cross-context synthesis (single-environment result)

### Part IV: Project D -- Language Interface
- Template-based language interface for the Monitor
- 7 templates for failure mode reporting
- Not central to v3.0 (orthogonal to Monitor training signal)

### Part V: Project E -- Neuro-Symbolic Verification
- DLR (Differentiable Logic Reasoner) cross-agent predicates
- The verified 1 positive result in the 11-comparison record
- v8 dlr_only: +0.06 at n=100, Bonferroni-corrected p=0.0433
- **Y5 connection**: Proposition 3 (Monitor + DLR hybrid) is the open
  prediction; P3 hybrid test is the pre-registered execution

### Part VI: Project F -- Multi-Agent (Sketch)
- Y3 paper (v1.0): cooperative MARL, 6-pathway investigation
- 5 of 6 pathways REFUTED at p<0.05
- The 1 positive is v8 dlr_only (not Monitor)
- **Y5 connection**: Condition 1 (distribution match) is violated in MARL
  due to joint critic training; framework predicts REFUTATION (correct)

### Part VII: Cross-Environment and Transfer
- DLR predicate transfer across 4 environments (H3, n=12, 3 seeds each)
- Y1 cross-environment validation
- **v3.0 addition**: 11-comparison cross-context record (RL -> MARL -> LLM)
  with the 6-method meta-analysis as a single table

### Part VIII: Discussion and Future Work
- The 4 sub-projects (A, C, D, E, F) in the v2.0 era
- v3.0 addition: synthesis of all 4 sub-projects with the Y5 Section 7.6
  framework
- Future work: P3 hybrid test, R4 (Monitor at 7B / 70B LLM scale)

### Part IX: Project G -- LLM Self-Monitoring (Y4, 2026-07-29)
- Y4 paper (v1.0): LLM self-monitoring H10
- 4 sample sizes: n=5, n=20, n=100, n=20 GSM8K 200-tok
- All 4 REFUTED at p > 0.05
- Pre-reg kill switch STOP-PAPER-REFUTED-REVERSE
- **Y5 connection**: Condition 2 (failure observability) is weakly
  violated in LLM context; AUROC ~ 0.50-0.65 (near chance)

### Part X: Cross-Context Synthesis (Y5 v1.3 master synthesis, NEW in v3.0)
- The 11-comparison cross-context record
- Y5 Section 7.6: formal framework (3 Convergence Conditions + 4 Propositions + 4 Refutations)
- Y5 Section 7.5.5: first-principles motivation (covariate shift + mutual information + PAC-learning)
- Y5 Section 7.6.2: Assumption A1 (positive mutual information)
- Y5 Section 7.6.3: logical disjunction of R1-R4 + cost-weighted observation
- Y5 Section 7.6.6: formal Monotonicity Lemma (16-cell Boolean lattice)
- Y5 Section 5.3.1 + 5.3.2: cross-task meta-analysis (6 methods)
- Y5 Section 8.5: deployment patterns (4 patterns)
- Y5 Section 9.6: framework limitations (3 explicit limitations)
- Y5 Appendix C: H1-H10 status table (post Y4 v0.6.1)
- Y5 Section 5.3.1: forest plot visualization

---

## Changes from v2.0 to v3.0

| Aspect | v2.0 (2026-07-30) | v3.0 (2026-08-01) |
|---|---|---|
| Y1 paper | draft | v1.0 (28 pages, +39.5) |
| Y3 paper | draft | v1.0 (17 pages, 5/6 REFUTED) |
| Y4 paper | draft | v1.0 (PDF, 4/4 REFUTED) |
| Y5 paper | absent | v1.3 (89 pages, master synthesis) |
| Sections | 9 | 10 (new Part X) |
| Framework | informal convergence-conditions | formal Section 7.6 with 4 named Refutations |
| Meta-analysis | absent | 6 methods (Fisher + Stouffer + Bonferroni + Holm + Hedges g + forest plot) |
| Bibliography | 13 entries | 20 entries (+7) |
| R1-R4 Refutations | implicit | 4 named with cost-weighted observation table |
| Monotonicity | absent | formal 16-cell Boolean lattice + proof |
| Deployment patterns | absent | 4 concrete patterns with failure modes |
| COLM 2026 cover letter | absent | v1.3 (15.6 KB) |
| Reviewer simulators | 1 (v1.0) | 3 (v1.0 / v1.2 / v1.3) |
| OpenReview package | absent | built (tar.gz + SHA-256 + checklist) |
| Pre-Regs | 3 (Y1, Y4 Amend 1+Add) | 4 (added P3 hybrid) |

## Build status (2026-08-01)

- v3.0 .tex source: COMMITTED (338 KB, 8794 lines)
- v3.0 PDF build: BLOCKED on v2.0 \\hline position error (line 222)
- v3.0 abstract: COMMITTED to thesis_v3_0_abstract.md (1.6 KB)
- v3.0 PDF build path: fix v2.0 \\hline, re-run pdflatex, then re-validate

## Cited pre-registered experiments

- Y1 v0.x: 15 seeds, LunarLander-v3
- Y3 v0.x: 6 pathways, ~150 seeds total
- Y4 v0.6.1: 4 sample sizes + 2 task families, 155 jobs (n=5 + n=20 + n=100 + n=20 GSM8K)
- P3 hybrid (in progress): 60 jobs, n=20 x 3 arms x 200 updates (tightened)
- P3 hybrid (full pre-reg): 300 jobs, n=100 x 3 arms x 800 updates (planned 2026-08-01 to 2026-08-15)

## Companion papers (all v1.0)

- papers/y1_paper_draft.md (Y1 v1.0 source)
- papers/monitor_signal_vs_dlr_6pathway.md (Y3 v1.0 source)
- papers/project_g_v0_5_h10_paper.md (Y4 v1.0 source)
- papers/y5_monitor_transfer_synthesis.md (Y5 v1.3 source)
- papers/y{1,3,4,5}_v*_*.{html,docx,pdf} (rendered outputs)

## References (7 new in v3.0)

- Shimodaira 2000 (covariate shift)
- Cover & Thomas 1991 (information theory)
- Valiant 1984 (PAC-learning)
- Haussler 1990 (PAC sample complexity)
- Hanley-McNeil 1982 (AUROC standard error)
- Holm 1979 (Bonferroni-Holm step-down)
- Hedges 1981 (bias-corrected d)

## Submission status

- COLM 2026 (v1.3): ready, tar.gz built, awaiting arXiv ID for final cover letter
- AAMAS 2027 (Y3): separate submission
- arXiv: package built, awaiting user ARXIV_TOKEN

## Funding and acknowledgments

This research is independently funded. Compute was provided by local
consumer hardware (NVIDIA RTX 4090 + Intel i9-13900K, 64 GB RAM). The
Archimedes Project is part of the AGI-2026-001 research initiative.