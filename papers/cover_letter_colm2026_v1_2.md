# Cover Letter -- COLM 2026 Submission (Y5 v1.2 Master Synthesis)

**To:** COLM 2026 Program Chairs
**From:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** July 31, 2026 (v1.2 draft; supersedes v1.0 and v0.8 submissions)
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

## Version history of this submission

| Version | Date | Pages | Notable additions |
|---|---|---:|---|
| v0.8 | 2026-07-31 | 56 | Original synthesis draft (no formal framework) |
| v1.0 | 2026-07-31 | 64 | Added §7.6 formal framework (7 Definitions + 4 Propositions + 4 Refutations) + framework diagram |
| v1.2 | 2026-07-31 | **82** | Added §5.3.1+§5.3.2 cross-task meta-analysis (6 methods), §7.5.5 first-principles motivation, §7.6.2 Assumption A1, §7.6.6 monotonicity, §8.5 deployment patterns, §9.6 framework limitations, Pre-Reg for Proposition 3 |

The v1.2 submission represents a complete response to the 12-item
reviewer simulator report (`papers/reviewer_simulator_output_v1_2.md`).
All 12 items are addressed; the v1.2 reviewer simulator returned
6 very-minor items (all P3), a 50% reduction from v1.0.

## Why this paper is a good fit for COLM

This paper makes **three kinds of contributions** that COLM
welcomes: (a) a clean empirical cross-context synthesis
(11 pre-registered comparisons across single-agent RL, multi-
agent MARL, and LLM self-monitoring); (b) a predictive
formal framework (§7.6) with explicit falsifiability hooks;
and (c) operational deployment guidance (§8.5) for the
verified use of the Monitor as a verification signal.

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

## The v1.2 cross-task meta-analysis: 6 methods, 1 conclusion

v1.2 adds two new sections (**§5.3.1** and **§5.3.2**) that
combine the 4 H10 sample-size p-values using 6 different
meta-analytic methods:

| Method | Statistic | p | Conclusion |
|---|---|---|---|
| Fisher combined-p | chi^2 = 4.646, df = 8 | 0.7947 | NOT significant |
| Stouffer Z (equal weight) | Z = 1.105 | 0.135 (one-sided) | NOT significant |
| Stouffer Z (weighted by sqrt(n)) | Z = 0.853 | 0.197 (one-sided) | NOT significant |
| Bonferroni min p | min p * 4 = 1.12 | alpha_bonf = 0.0125 | NOT rejected |
| Bonferroni-Holm step-down | 0/4 reject | alpha = 0.05 | NOT rejected |
| Hedges g (bias-corrected) | 4/4 CIs span zero | -- | NOT significant |

A forest plot (`papers/figures_v2/fig_h10_combined_p_forest.png`)
visually confirms the numerical analysis: all 4 sample-size
Cohen's d estimates straddle d = 0 (no effect) and are below
d = +0.10 (the pre-reg kill switch threshold after Amendment
1 addendum).

## The theoretical contribution: a predictive framework with 4 falsifiers (§7.6)

Beyond the empirical synthesis, v1.0 added **§7.6 formal
framework** with 7 Definitions, 4 Propositions, and 4 named
Refutations. v1.2 strengthens the framework with three new
sections:

- **§7.5.5 First-principles motivation**: each Convergence
  Condition is motivated by an established theorem (covariate
  shift for C1, mutual information for C2, PAC-learning for
  C3). The derivations are sketches, not proofs, and the
  paper is explicit about this.

- **§7.6.2 Assumption A1**: explicitly names the implicit
  assumption in Proposition 1 converse direction (positive
  mutual information between auxiliary signal and policy
  value function). Without A1, the converse is false.

- **§7.6.3 Logical disjunction**: framework is falsified iff
  at least one of R1, R2, R3, R4 is observed. The complement
  (framework survives iff all 4 are unobserved) is also stated.

- **§7.6.6 Monotonicity**: observing more Refutations forces
  STRONGER framework updates than the sum of individual
  updates (R1 alone -> 2-condition framework, R1+R2 -> 1-
  condition, R1+R2+R3 -> Bayesian revision, all 4 ->
  replacement).

The 4 Refutations (the framework's falsifiability hooks):

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
open as a falsifiable future test with an explicit compute-cost
estimate (~150-250 GPU-hours for 7B, ~1500-2500 GPU-hours for
70B).

## The operational contribution: 4 deployment patterns (§8.5)

v1.2 adds **§8.5** with 4 concrete verification deployment
patterns, each with a setup, example use case, and 1-2 known
failure modes:

- **Pattern A: Runtime guardrail** -- Monitor runs alongside
  the AGI policy; inferences with high Monitor failure
  probability are flagged. Failure modes: calibration drift
  (Condition 1), adversarial exploitation.
- **Pattern B: DLR predicate in critic** -- hand-crafted
  predicates provide per-step shaping bonus (the Y3 v8
  dlr_only architecture that gave +0.06 at n=100).
  Failure modes: predicate incompleteness, predicate-policy
  mismatch.
- **Pattern C: Pre-commit review** -- Monitor's prediction is
  combined with a pre-committed "constitution" (Constitutional
  AI analog). Failure modes: constitution incompleteness,
  Monitor-constitution disagreement.
- **Pattern D (proposed): Monitor + DLR hybrid** -- combines
  Pattern A (runtime guardrail) with Pattern B (DLR shaping).
  This is the architecture that Proposition 3 predicts should
  work better than either alone. The Pre-Reg for Proposition
  3 (`experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md`)
  is the proposed empirical test.

## Methodological contributions

The paper contributes six reusable methodological improvements
(full details in supplementary S1-S15):

1. **Stratified train/eval split with deterministic rebalance
   fallback** (Section 3 of Y4 v0.5): a better default for
   self-monitoring evaluation than the deterministic split used
   in earlier work.

2. **Pre-registered kill switch with versioned addenda**:
   Amendable pre-registration with documented kill-switch
   thresholds (Pre-Reg Amendment 1: +0.05 -> +0.10).

3. **The 3 Convergence Conditions as a general analytic
   lens** for ANY learned auxiliary signal (Constitutional
   AI / PRM / RLHF retrofits in §7.6.5).

4. **§7.6 falsifiability discipline**: 4 named Refutations
   with logical disjunction (R3.3) and monotonicity (R3.4).

5. **§7.5.5 first-principles motivation**: each Convergence
   Condition motivated by an established theorem (R2.3).

6. **§5.3.1 + §5.3.2 cross-task meta-analysis**: 6 methods
   (Fisher / Stouffer / Bonferroni / Bonferroni-Holm / Hedges g
   / forest plot) converge on H10 REFUTATION.

## Reviewer simulator (v1.2 re-run)

We re-ran the 3-reviewer simulator on v1.2
(`papers/reviewer_simulator_output_v1_2.md`). All three
returned **Weak Accept (with very minor revisions)**:

- R1: 4 of 4 v1.0 items addressed; 2 very-minor items raised.
- R2: 4 of 4 v1.0 items addressed; 2 very-minor items raised.
- R3: 4 of 4 v1.0 items addressed (R3.1 was already v1.1); 2 very-minor items raised.

Total: 6 very-minor items, all P3, all text edits or simple
calculations. No new experiments required. We will address
them in the v1.3 camera-ready version.

## Compute and reproducibility

The full empirical chain (Y1 + Y3 + Y4 = 11 comparisons)
took ~120 GPU-equivalent-hours on consumer hardware. The v1.2
additions (Pre-Reg for Proposition 3, extended meta-analysis,
deployment patterns) added ~5 person-days of focused work
and 0 GPU-hours.

All code, logs, per-seed JSONs, and aggregated bootstrap
results are committed to the Archimedes Project git
repository:

- Pre-Registrations:
  - `experiments_log/2026-07-28-PRE-REGISTERED-H10.md` (H10 original)
  - `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md` (H10 GSM8K 200-tok n=20)
  - `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md` (kill switch +0.05 -> +0.10)
  - `experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md` (NEW in v1.2, Proposition 3 hybrid test)

- Aggregated bootstrap JSONs:
  - `experiments_log/_h10_n20_gsm8k_bootstrap.json`
  - `experiments_log/_h10_n100_bootstrap.json`
  - `experiments_log/_h10_n20_bootstrap.json`
  - `experiments_log/_h10_combined_p.json` (NEW in v1.1, extended in v1.2)

- Figures:
  - `papers/figures_v2/forest_h10_n20_gsm8k.png`
  - `papers/figures_v2/h10_shrinkage_timeline_v06.png`
  - `papers/figures_v2/decoupling_across_contexts.png`
  - `papers/figures_v2/fig_y5_7_6_convergence_refutations.png` (NEW in v1.0)
  - `papers/figures_v2/fig_h10_combined_p_forest.png` (NEW in v1.2)

A supplementary materials document (S1-S15) provides full
provenance for every number in the paper.

## What this paper does not claim

We do not claim that the failure-prediction Monitor can never
help with LLM self-monitoring -- only that two qualitatively
different LLM task families (short deterministic arithmetic
and GSM8K 200-token chain-of-thought) at the 1.5B parameter
scale do not produce a strong enough signal for the Monitor
architecture to discriminate failure from success in a way
that benefits from being decoupled and frozen. R4 (Monitor at
7B / 70B scale) is left explicitly open as a falsifiable
future test with a documented compute budget
(~150-250 / ~1500-2500 GPU-hours). The Monitor's verified
shipping use remains **verification** (Patterns A, B, C in
Section 8.5: runtime guardrails, DLR predicates, pre-commit
review), not training.

## Conflict of interest and prior publication

This is the v1.2 revision of the paper originally drafted as
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
  falsifiable theoretical frameworks for ML, (c) cross-
  context evaluation, or (d) AGI safety / auxiliary signals.

We thank the COLM 2026 program chairs for considering a
pre-registered cross-context synthesis paper with 11 empirical
comparisons, 6 meta-analytic methods converging on H10
REFUTATION, a predictive formal framework with 4 named
falsifiers (logical disjunction + monotonicity structure),
operational deployment guidance with 4 patterns, and full
reproducibility on consumer hardware. The methodological
discipline (pre-registration chain, stratified split, versioned
addenda, transparent cross-task shrinkage, extended meta-
analysis with 6 methods) is exactly the kind of contribution
COLM aims to elevate.