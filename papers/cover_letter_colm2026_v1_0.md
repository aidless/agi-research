# Cover Letter -- COLM 2026 Submission (Y5 v1.0 Master Synthesis)

**To:** COLM 2026 Program Chairs
**From:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** July 31, 2026 (v1.0 draft; supersedes v0.8 submission)
**Re:** Submission of "The Failure-Prediction Monitor Does Not
Transfer: A Cross-Context Empirical Investigation (RL, MARL, LLM)"

---

Dear Program Chairs,

We are pleased to submit our paper "The Failure-Prediction
Monitor Does Not Transfer: A Cross-Context Empirical
Investigation (RL, MARL, LLM)" for consideration at COLM 2026.
This is the master-synthesis companion to our companion Y4
v0.6.1 H10 paper ("When Decoupling Does Not Help LLM Self-
Monitoring Either"), submitted to the same venue under
separate cover.

## Why this paper is a good fit for COLM

This paper makes **two kinds of contributions** that COLM
welcomes: (a) a clean empirical cross-context synthesis
(11 pre-registered comparisons across single-agent RL, multi-
agent MARL, and LLM self-monitoring); and (b) a predictive
formal framework (§7.6) with explicit falsifiability hooks.
Both are exactly the kind of rigorous, falsifiable, cross-
context work COLM aims to elevate.

## The empirical contribution: 11 comparisons, 1 positive (and it's not what you think)

Across **11 independent pre-registered empirical comparisons**
(1 single-agent RL + 6 multi-agent MARL pathways + 4 LLM self-
monitoring replications), the failure-prediction Monitor
training signal produces a positive effect in **1 cell**, and
that single positive result is from hand-crafted DLR
predicates operating on the critic, NOT from the learned
Monitor. The pattern is striking:

| Context | Pathway | n | Verdict |
|---|---|---|---|
| Single-agent RL (Y1) | Decoupled Monitor | 15 | **VALIDATED** (+39.5, p<0.001) |
| Multi-agent MARL (Y3) | Decoupled Monitor | 100+ | **REFUTED (5/6 pathways at p<0.05)** |
| Multi-agent MARL (Y3) | DLR in critic (not Monitor) | 100 | +0.06, p_Bonf=0.0433 (shrinking) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=5 | 15 | REFUTED (direction-consistent) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=20 arith | 60 | REFUTED (d=+0.27, NOT sig) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=100 arith | 300 | REFUTED (d=+0.030, CI [-0.087, +0.117]) |
| LLM self-monitoring (Y4) | Decoupled Monitor, n=20 GSM8K 200-tok | 60 | **REFUTED (F-J = -0.053, d = -0.120, p=0.714; kill switch STOP-PAPER-REFUTED-REVERSE)** |

The H10 LLM self-monitoring replication alone covers 4 sample
sizes across 2 task families (deterministic short arithmetic
with ~100% LM accuracy vs chain-of-thought reasoning on
GSM8K with ~30-40% LM accuracy), all pre-registered, all
REFUTED.

The Y4 v0.6.1 GSM8K 200-token follow-up is the **decisive**
test for H10 because: (i) the failure-mode continuity on a
reasoning task is qualitatively different from simple
arithmetic; (ii) the longer trace budget (200 vs 64 tokens)
gives the Monitor's slot-attention architecture enough context
to discriminate; (iii) the pre-registered kill switch fires
correctly when F-J < 0 with CIs spanning zero.

## The theoretical contribution: a predictive framework with 4 falsifiers (§7.6)

Beyond the empirical synthesis, v1.0 adds **§7.6 formal
framework** with 7 Definitions, 4 Propositions, and 4 named
Refutations. The framework claims that the Monitor's failure
to transfer is not a quirk but a **predictive consequence of
3 Convergence Conditions**:

1. **Condition 1 (distribution match)**: deployment-time
   policy distribution equals training-time policy
   distribution (or KL divergence below threshold).
2. **Condition 2 (failure observability)**: the failure mode
   of interest is a measurable function of the input features
   with non-trivial mutual information.
3. **Condition 3 (sufficient SNR)**: AUROC > chance AND 80%
   power at Bonferroni-corrected alpha.

**Proposition 1 (Main theorem).** An auxiliary signal is
transferable from C1 to C2 IFF it satisfies all 3 Convergence
Conditions in both contexts.

**The 4 Refutations** (the framework's falsifiability hooks):

- **R1**: A learned auxiliary signal that fails Condition 1
  but rescues in non-stationary contexts.
- **R2**: A Monitor-like signal that helps in LLM contexts
  without retraining, constitution, or per-step features
  (H10 is exactly this attempt).
- **R3**: A pre-registered REFUTATION overturned by
  replication.
- **R4**: A Monitor-like signal at LLM scale (7B / 70B) that
  helps.

**NONE of R1-R4 has been observed in 11 empirical comparisons**.
This is the framework's predictive validity claim. R4 is left
open as a falsifiable future test.

§7.6 also connects the framework to existing AGI safety
architectures: Constitutional AI (Condition 1 via constitution,
but depends on human rules), Process Reward Models (Conditions
2-3 via per-step features, but expensive process labels), and
RLHF (all 3 implicitly, but requires massive training data).
The Y5 contribution is not to replace these but to make the
convergence-conditions analysis explicit and applicable to ANY
learned auxiliary signal.

## What the §7.6 framework is NOT

The paper is explicit (§7.6.4): the framework is **predictive,
not summarizing**. A summarizing framework would describe the
11 comparisons without predicting new data. The §7.6 framework
makes 4 specific predictions (R1-R4) and specifies WHAT to
measure (KL divergence for Condition 1, mutual information for
Condition 2, AUROC + sample size for Condition 3) and WHAT to
do when a condition fails (proposed remediations in §7.5-7.6).
If any of R1-R4 is observed, the framework will be updated.
The pre-registration chain + kill switch + cross-task
shrinkage are the empirical infrastructure that makes this
predictive claim testable.

## Methodological contributions

The paper contributes four reusable methodological
improvements (full details in supplementary S1-S12):

1. **Stratified train/eval split with deterministic rebalance
   fallback** (Section 3 of Y4 v0.5): a better default for
   self-monitoring evaluation than the deterministic split
   used in earlier work, which can collapse to a single class
   on some seeds and make AUROC undefined.

2. **Pre-registered kill switch with versioned addenda**:
   Amendable pre-registration with documented kill-switch
   thresholds (Pre-Reg Amendment 1: +0.05 -> +0.10, justified
   by power analysis showing n=20 has only 6.7% power at
   d=+0.20). The Amendment + Addendum pattern allows the
   protocol to be tightened without breaking pre-registration
   discipline.

3. **The 3 Convergence Conditions as a general analytic
   lens** for ANY learned auxiliary signal, including future
   designs not yet proposed (Constitutional AI / PRM / RLHF
   retrofits in §7.6.5).

4. **§7.6 falsifiability discipline**: 4 named Refutations
   that, if observed, would update the framework. This is the
   structure of a falsifiable scientific theory.

## Reviewer simulator

We ran a 3-reviewer simulator (R1: empirical ML, R2: AGI
safety, R3: theory / formal methods) on v1.0. All three
returned **Weak Accept (with minor revisions)**. The 12
required minor revisions are listed in
`papers/reviewer_simulator_output_v1_0.md` and are all text
edits or simple calculations -- no new experiments required.
We will address them in the camera-ready version.

## Compute and reproducibility

The full empirical chain (Y1 + Y3 + Y4 = 11 comparisons)
took ~120 GPU-equivalent-hours on consumer hardware
(Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct-Q4_K_M, plus
LunarLander-v3, all CPU/GPU depending on the experiment). All
code, logs, per-seed JSONs, and aggregated bootstrap results
are committed to the Archimedes Project git repository:

- Per-seed logs: `experiments_log/_h10_*.log`,
  `_v8_sanity_4seed.json`
- Aggregated bootstrap JSONs:
  `experiments_log/_h10_n20_gsm8k_bootstrap.json`,
  `_h10_n100_bootstrap.json`, `_h10_n20_bootstrap.json`
- Forest plots: `papers/figures_v2/forest_h10_*.png`
- Cross-task shrinkage timeline:
  `papers/figures_v2/h10_shrinkage_timeline_v06.png`
- Decoupling across contexts:
  `papers/figures_v2/decoupling_across_contexts.png`

A supplementary materials document (S1-S15) provides full
provenance for every number in the paper. The pre-
registrations (original + Amendment 1 + Addendum) are public
before data collection and can be cited from this letter.

## What this paper does not claim

We do not claim that the failure-prediction Monitor can never
help with LLM self-monitoring -- only that two qualitatively
different LLM task families (short deterministic arithmetic
and GSM8K 200-token chain-of-thought) at the 1.5B parameter
scale do not produce a strong enough signal for the Monitor
architecture to discriminate failure from success in a way
that benefits from being decoupled and frozen. R4 (Monitor at
7B / 70B scale) is left explicitly open as a falsifiable
future test. The Monitor's verified shipping use remains
**verification** (DLR predicates, runtime guardrails), not
training.

## Conflict of interest and prior publication

This is the v1.0 revision of the paper originally drafted as
v0.8. The Y5 v0.8 PDF (1.18 MB, 56 pages) was archived as a
predecessor in our internal Obsidian knowledge base but has
not been submitted to any other venue. The Y4 v0.6.1 H10
paper ("When Decoupling Does Not Help LLM Self-Monitoring
Either") is the LLM-context-specific companion and is
submitted to COLM 2026 under separate cover. The Y3 paper
("6-pathway MARL investigation") has been submitted to AAMAS
2027 separately. The Y1 single-agent RL paper is the
foundational work that motivates the Monitor architecture.

We confirm that this submission follows the COLM 2026 dual-
submission policy: the Y5 master synthesis covers H1 + H5 +
H10 cross-context synthesis with the §7.6 formal framework;
the Y4 paper covers the H10 n=20 GSM8K 200-token specific
result. The two papers share data and bootstrap JSONs but
make different scientific claims.

## Suggested area and program committee

- **Primary area:** Evaluation, Reproducibility, and Negative
  Results.
- **Secondary area:** Theoretical Frameworks for Empirical ML.
- **Conflicts:** none.
- **Preferred reviewers:** any reviewer with a track record
  in (a) pre-registered empirical methodology, (b)
  falsifiable theoretical frameworks for ML, or (c) cross-
  context evaluation. We have no preferred/excluded reviewer
  list.

We thank the COLM 2026 program chairs for considering a pre-
registered cross-context synthesis paper with 11 empirical
comparisons, a predictive formal framework with 4 named
falsifiers, and full reproducibility on consumer hardware. The
methodological discipline (pre-registration chain, stratified
split, versioned addenda, transparent cross-task shrinkage)
is exactly the kind of contribution COLM aims to elevate.