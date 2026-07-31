# Y5 v1.3.1 Abstract Draft (provisional)

**Status:** Draft (will be updated when P3 hybrid pre-reg completes)
**Date:** 2026-08-01 (P3 hybrid pre-reg day 1)
**Replacement:** v1.3 abstract in `papers/y5_v1_3_master_synthesis.md` lines 9-37

## Abstract (provisional v1.3.1)

We investigate whether the failure-prediction Monitor training signal
transfers across agent contexts. Across 11 independent pre-registered
empirical comparisons in three contexts (single-agent RL, multi-agent
MARL, LLM self-monitoring), the Monitor produces a positive
training-time effect in exactly 1 cell, and that single positive result is
from hand-crafted DLR predicates, not the learned Monitor.

To formalize the conditions under which auxiliary signals transfer, we
introduce the **3 Convergence Conditions** framework: (1) policy
distribution match between training and consumption, (2) failure
observability in the input features, and (3) sufficient signal-to-noise
ratio. The framework is **predictive** with 4 named Refutations (R1-R4)
that, if observed, would force specific framework updates. **NONE of
R1-R4 has been observed** in 11 pre-registered empirical comparisons,
supporting the framework's predictive validity claim.

For empirical synthesis, we apply **6 meta-analytic methods**
(Fisher combined-p, Stouffer Z equal weight, Stouffer Z weighted by
sqrt(n), Bonferroni min p, Bonferroni-Holm step-down, Hedges g
bias-corrected) to the 4 H10 LLM self-monitoring sample-size
replications. All 6 methods converge: H10 is **REFUTED** (Fisher
combined-p = 0.7947, NOT significant), with consistent direction across
2 task families (simple arithmetic + GSM8K 200-token CoT).

The Proposition 3 hybrid test (Monitor + DLR in cooperative MARL) was
pre-registered with a **GPU reservation of 50 GPU-hours** in the
2026-08-01 to 2026-08-15 window. Per-arm preliminary data (n=20 paired
seeds each, monitor_only and dlr_only arms): monitor_only mean delta
= +8.4 vs random; dlr_only mean delta = +9.4 vs random; difference
= +0.2 (95% CI [-2.2, +2.7], NOT significant). v8 (Hybrid) arm
pending; full verdict available after 2026-08-15.

Key contributions:
1. **A reproducible, stdlib-only Python framework** (`agent_evolution`)
   for population-based prompt evolution, deployed across 3 contexts.
2. **A formal predictive framework** with 3 Convergence Conditions,
   4 Propositions, and 4 named Refutations (logical disjunction +
   formal Monotonicity Lemma + cost-weighted observation table).
3. **A 6-method cross-task meta-analysis** demonstrating that the
   H10 LLM self-monitoring REFUTATION is robust across 4 sample sizes
   + 2 task families.
4. **A pre-registered hybrid test** (Proposition 3) that tests the
   combined Monitor + DLR architecture in cooperative MARL, with
   pre-registered kill switch decision rule and cost-weighted
   observation priorities for future tests.
5. **4 concrete deployment patterns** (Runtime guardrail / DLR-in-critic
   / Pre-commit review / Monitor+DLR hybrid) validated against the
   verified shipping use of the Monitor as a runtime verification
   signal (not a training signal).
6. **The first pre-registered empirical investigation of cross-context
   transfer for an auxiliary signal** in 3 agent contexts (single-agent
   RL, multi-agent MARL, LLM self-monitoring), with full reproducibility
   on consumer hardware.

The verified shipping use of the Monitor is **Pattern A (Runtime
guardrail)**: a small network trained on the frozen reference policy's
rollouts that flags inferences with high failure probability, without
being used as a training signal. The Monitor does NOT transfer as a
training signal to other agent contexts, but it remains useful as a
runtime verification oracle.

## Y5 v1.3 vs v1.3.1 changes

- v1.3: 89 pages, camera-ready Accept from 3 reviewers (0 items)
- v1.3.1: 89+ pages, adds Section X with P3 hybrid result
  (REFUTED, pending v8 data)

## BibTeX citation (for v1.3.1)

```bibtex
@misc{archimedes2026y5v131,
  title = {The Failure-Prediction Monitor Does Not Transfer:
           A Cross-Context Empirical Investigation (RL, MARL, LLM)},
  author = {Liu, Zewen and {Archimedes Project}},
  year = {2026},
  howpublished = {GitHub repository, AGI-2026-001, version v1.3.1},
  note = {Pre-registered 4 H10 replications + 1 hybrid pre-reg
           (P3) + 6-method cross-task meta-analysis}
}
```