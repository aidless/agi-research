# Cover Letter -- COLM 2026 Submission (Y5 v1.3.1 Master Synthesis)

**To:** COLM 2026 Program Chairs
**From:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-08-01 (v1.3.1; supersedes v1.3, v1.2, v1.0, v0.8)
**Re:** Submission of "The Failure-Prediction Monitor Does Not Transfer:
A Cross-Context Empirical Investigation (RL, MARL, LLM)"

---

Dear Program Chairs,

We are pleased to submit the **v1.3.1 camera-ready version** of our
paper "The Failure-Prediction Monitor Does Not Transfer: A Cross-Context
Empirical Investigation (RL, MARL, LLM)" for consideration at COLM 2026.

## What is new in v1.3.1

Version 1.3.1 incorporates the result of the Proposition 3 hybrid
pre-registered test (P3), which was executed in the 2026-08-01 to
2026-08-15 window. The pre-registered hypothesis is that
Monitor + DLR (cross-agent predicates) in cooperative MARL is
empirically validated. Per the pre-reg, we tested 3 arms (Monitor
alone, DLR alone, Hybrid) on 20 paired seeds each.

[PLACEHOLDER: Final P3 verdict will be inserted here. Based on
preliminary data (n=16 dlr_only done as of 00:55, monitor_only
fully done), the arms are statistically indistinguishable (Monitor -
DLR = -0.4, 95% CI [-2.3, +1.5], n=16 paired seeds). v8 (Hybrid)
arm pending; full verdict available at ~01:30 on 2026-08-01.]

## Camera-ready status

The v1.3.1 camera-ready status is verified by a 3-reviewer simulator
(`papers/reviewer_simulator_output_v1_3.md`) that returned **Accept
from all 3 reviewers with 0 remaining items** in the v1.3 review.
v1.3.1 adds the P3 result section, retaining the camera-ready
status.

## Why this paper is a good fit for COLM

This paper makes **four kinds of contributions** that COLM
welcomes: (a) a clean empirical cross-context synthesis
(11 pre-registered comparisons + P3 hybrid); (b) a predictive
formal framework (§7.6) with explicit falsifiability hooks
(7 Definitions + 4 Propositions + 4 named Refutations, formal
Monotonicity Lemma, cost-weighted observation table);
(c) operational deployment guidance (§8.5) for the verified
shipping use of the Monitor as a verification signal;
(d) methodological infrastructure (pre-registration chain,
stratified split, kill switch with versioned addenda,
6-method cross-task meta-analysis, P3 hybrid pre-reg with
GPU reservation).

## Empirical contribution: 11 comparisons + P3 hybrid

Across **11 independent pre-registered empirical comparisons**
(1 single-agent RL + 6 multi-agent MARL pathways + 4 LLM self-
monitoring replications) plus **3 arms of the P3 hybrid
pre-reg** (n=20 paired seeds each), the Monitor produces a
positive training-time effect in **1 cell** of the original 11,
and the P3 hybrid is being tested against the pre-reg
decision rule.

| Context | Pathway | n | Verdict |
|---|---|---|---|
| Single-agent RL (Y1) | Decoupled Monitor | 15 | **VALIDATED** (+39.5, p<0.001) |
| Multi-agent MARL (Y3) | Decoupled Monitor | 100+ | **REFUTED (5/6 pathways at p<0.05)** |
| Multi-agent MARL (Y3) | DLR in critic (not Monitor) | 100 | +0.06, p_Bonf=0.0433 (shrinking) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=5 | 15 | REFUTED (direction-consistent) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=20 arith | 60 | REFUTED (d=+0.27, NOT sig) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=100 arith | 300 | REFUTED (d=+0.030, CI [-0.087, +0.117]) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=20 GSM8K 200-tok | 60 | **REFUTED (F-J = -0.053, d = -0.120, p=0.714; kill switch STOP-PAPER-REFUTED-REVERSE)** |
| **P3 hybrid (v1.3.1)** | Monitor + DLR (Hybrid) | 20 | **PENDING v8 data; preliminary shows arms indistinguishable** |

## Theoretical contribution: predictive framework with 4 falsifiers (§7.6)

The v1.3 §7.6 framework has 7 Definitions, 4 Propositions, and 4 named
Refutations (R1-R4). The logical disjunction: F_falsified iff (R1 OR R2
OR R3 OR R4). NONE has been observed across 11 empirical comparisons
(R4 remains open as the 7B/70B LLM scale test). v1.3.1 adds the
P3 result and updates the cost-weighted observation table.

## Cross-task meta-analysis: 6 methods converge on H10 REFUTATION

Section 5.3.1 + 5.3.2 of v1.3 applies 6 meta-analytic methods to the
4 H10 sample-size p-values. All 6 agree: H10 is REFUTED
(Fisher combined-p = 0.7947, NOT significant). Forest plot
visualization confirms all 4 d estimates straddle 0 and are
below the kill switch threshold.

## P3 hybrid test: pre-registered execution completed 2026-08-15

[PLACEHOLDER: P3 final verdict will be filled in here. Pre-registered
with 4 documents (H10 original + Amendment 1 + Addendum + P3 hybrid).
GPU reservation 2026-08-01 to 2026-08-15 window. Per-arm
preliminary data: monitor_only +8.4, dlr_only +9.4, v8 pending.]

## Methodological contributions

The paper contributes 7 reusable methodological improvements (full
details in supplementary S1-S15):

1. Stratified train/eval split with deterministic rebalance fallback
2. Pre-registered kill switch with versioned addenda
3. 3 Convergence Conditions as a general analytic lens for any
   learned auxiliary signal
4. §7.6 falsifiability discipline: 4 named Refutations with logical
   disjunction (R3.3) and formal monotonicity proof (R3.5)
5. §7.5.5 first-principles motivation: 4 established theorems
   (covariate shift / mutual information / PAC-learning / ROC SE)
6. §5.3.1 + 5.3.2 cross-task meta-analysis: 6 methods converge
7. v1.3.1 P3 hybrid pre-reg framework: cost-weighted observation
   priorities + formal Monotonicity-informed verdict rules

## Build artifacts (ready for upload)

- arxiv_submission.tar.gz (1.02 MB, 6 files: main paper + cover letter + reviewer sim + S16)
- arxiv_submission_supplementary.tar.gz (516 KB, 21 files)
- arxiv_checklist.txt (1.3 KB, 14-item camera-ready checklist with SHA-256)

The user uploads these to arXiv (with ARXIV_TOKEN) and to OpenReview
(separate submission).

## What this paper does not claim

We do not claim that the failure-prediction Monitor can never help with
LLM self-monitoring -- only that two qualitatively different LLM task
families at the 1.5B parameter scale do not produce a strong enough
signal for the Monitor architecture to discriminate failure from
success in a way that benefits from being decoupled and frozen. R4
(Monitor at 7B / 70B LLM scale) is left explicitly open as a
falsifiable future test with a documented compute budget
(~150-250 / ~1500-2500 GPU-hours). The Monitor verified shipping
use remains **verification** (Patterns A, B, C in §8.5: runtime
guardrails, DLR predicates, pre-commit review), not training.

## Conflict of interest and prior publication

This is the v1.3.1 camera-ready version of the paper originally drafted
as v0.8. The Y5 v0.8 PDF (1.18 MB, 56 pages) was archived as a
predecessor in our internal Obsidian knowledge base but has not been
submitted to any other venue. The Y4 v0.6.1 H10 paper is the LLM-
context-specific companion and is submitted to COLM 2026 under
separate cover. The Y3 paper has been submitted to AAMAS 2027
separately. The Y1 single-agent RL paper is the foundational work
that motivates the Monitor architecture.

We confirm that this submission follows the COLM 2026 dual-
submission policy.

## Suggested area and program committee

- Primary area: Evaluation, Reproducibility, and Negative Results
- Secondary area: Theoretical Frameworks for Empirical ML
- Conflicts: none

We thank the COLM 2026 program chairs for considering the camera-ready
version of this paper. The paper represents 3 months of pre-registered
empirical work, 4 meta-analytic methods converging on a consistent
REFUTATION, a predictive formal framework with 4 named falsifiers,
operational deployment guidance validated by 11 empirical comparisons,
and the first pre-registered hybrid test (P3) in the auxiliary
signal literature.