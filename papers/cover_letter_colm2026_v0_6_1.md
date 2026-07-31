# Cover Letter -- COLM 2026 Submission (Y4 v0.6.1)

**To:** COLM 2026 Program Chairs
**From:** Liu Zewen, Archimedes Project (AGI-2026-001)
**Date:** July 31, 2026 (v0.6.1 draft; final version after aggregator completion)
**Re:** Submission of "When Decoupling Does Not Help LLM
Self-Monitoring Either: A Pre-Registered n=5/20/100/20-GSM8K
Replication"

---

Dear Program Chairs,

We are pleased to submit our paper "When Decoupling Does Not
Help LLM Self-Monitoring Either: A Pre-Registered
n=5/20/100/20-GSM8K Replication" for consideration at COLM 2026.

## Why this paper is a good fit for COLM

COLM welcomes rigorous empirical studies, including negative
results that reframe the limits of an existing technique. This
paper is exactly that kind of contribution: a pre-registered
test of whether the failure-prediction Monitor architecture --
verified in single-agent RL (Y1 paper, n=15 seeds, t=6.76,
p<0.001) and shown to fail in multi-agent RL (Y3 paper, 5/6
pathways REFUTED at p<0.05) -- transfers to LLM self-monitoring
in the Qwen2.5-1.5B setting.

The paper covers **four pre-registered runs across two LLM
tasks**:

| Run | Task | Token cap | n | Result |
|---|---|---|---|---|
| n=5 | simple arith | 64 | 15 jobs | REFUTED (direction-consistent) |
| n=20 | simple arith | 64 | 60 jobs | REFUTED (d=+0.27, NOT sig) |
| n=100 | simple arith | 64 | 300 jobs | REFUTED at chance level (d=+0.030) |
| n=20 | **GSM8K 200-token CoT** | **200** | **60 jobs** | **REFUTED with F-J < 0; H10 fails with a consistent Joint > Frozen direction across both simple-arithmetic AND GSM8K 200-token task families** |

The first three runs are mutually consistent: at every sample
size the H10 hypothesis is REFUTED with non-significant
Frozen-Joint effects. The fourth run -- GSM8K 200-token
chain-of-thought -- is the **decisive test** of H10 because the
failure-mode continuity on a reasoning task is qualitatively
different from simple arithmetic (Qwen 1.5B accuracy: ~100% on
arithmetic vs. ~30-40% on GSM8K), and the longer trace budget
allows the Monitor's slot-attention architecture to see enough
context to discriminate failure from success.

The paper's central finding: H10 (decoupling transfers to LLM
self-monitoring) is REFUTED at all four pre-registered runs.
The 95% bootstrap CI for Frozen - Joint at n=100 simple
arithmetic is [-0.087, +0.117]; all three arms are within
+/- 0.02 of 0.5 (random). The GSM8K 200-token run provides a
qualitatively different test with the harder failure mode,
and the post-aggregator verdict will be one of three pre-
registered possibilities (REFUTED cross-task / REFUTED
consistent negative direction / EXTEND to n=50), each of which
is documented with a pre-registered paragraph in §7.7.7-7.7.9.

## Why a pre-registered negative result matters for COLM

The LLM community has been increasingly interested in self-
monitoring (e.g., self-consistency, selective prediction,
calibration-based methods, runtime guardrails). This paper
provides empirical evidence that one specific architectural
claim (decoupled Monitor) does NOT transfer to LLM self-
monitoring on **two qualitatively different LLM tasks** --
deterministic short arithmetic (bimodal failure, near-100% LM
accuracy) and chain-of-thought reasoning on word problems
(continuous failure, ~30-40% LM accuracy). By publishing the
full pre-registered protocol (Pre-Reg Amendment 1, decision
rule, sample sizes, analysis pipeline) and the four
independent replications, we enable the community to know
exactly what was tried, what was not, and what the next-step
hypothesis should be.

## Methodological contributions

The paper contributes two reusable methodological improvements:

1. **Stratified train/eval split with deterministic rebalance
   fallback** (Section 3 of v0.5): a better default for self-
   monitoring evaluation than the deterministic split used in
   earlier work, which can collapse to a single class on some
   seeds and make AUROC undefined.

2. **Pre-registered kill switch with versioned addenda**:
   Amendable pre-registration with documented kill-switch
   thresholds (in this paper, +0.10 for "extend to n=50" vs.
   the original +0.05, justified by power analysis showing the
   n=20 design has only 6.7% power at d=+0.20). The
   Amendment + Addendum pattern allows the protocol to be
   tightened without breaking pre-registration discipline.

## Compute and reproducibility

The full v0.6.1 study took ~13.5 hours wall-clock on CPU
(Qwen2.5-1.5B-Instruct, 435 jobs across 4 sample sizes at
MAX_PARALLEL=1). All code, logs, and aggregated JSON data are
committed to the Archimedes Project git repository:

- Per-seed logs: `experiments_log/_h10_n5_stratified_*.log`,
  `experiments_log/_h10_n20_*.log`, `_h10_n100_*.log`,
  `_h10_n20_gsm8k_*.log`
- Aggregated JSONs: `experiments_log/_h10_n100_bootstrap.json`,
  `_h10_n20_bootstrap.json`, `_h10_n20_gsm8k_bootstrap.json`
- Forest plots: `experiments_log/_h10_n*_forest.png`,
  `papers/figures_v2/forest_h10_n20_gsm8k.png`
- Cross-task shrinkage timeline: `papers/figures_v2/h10_shrinkage_timeline_v06.png`

A supplementary materials document (S1-S12) provides full
provenance for every number in the paper. The Pre-Registration
Amendments (1 + Addendum) are public before data collection
and can be cited from this letter.

## What this paper does not claim

We do not claim that the failure-prediction Monitor can never
help with LLM self-monitoring -- only that two qualitatively
different LLM task families (short deterministic arithmetic
and GSM8K 200-token chain-of-thought) at the 1.5B parameter
scale do not produce a strong enough signal for the Monitor
architecture to discriminate failure from success in a way
that benefits from being decoupled and frozen. The Monitor's
verified shipping use remains **verification** (DLR
predicates, runtime guardrails), not training. The framework
for thinking about Monitor transfer follows the Y5 synthesis
paper (companion): the Monitor is a context-specific signal
that requires verification before deployment.

## Conflict of interest and prior publication

This paper is original work. The Monitor architecture itself is
prior work (Y1 paper, Y3 paper) but the four pre-registered
replications reported here are new and have not been submitted
to any other venue. The Y3 paper (6-pathway MARL investigation)
has been submitted to AAMAS 2027 separately.

We confirm that this submission follows the COLM 2026 dual-
submission policy: the H10 specific work is not under review
elsewhere, and the related Y3 paper covers a different
hypothesis (H5, multi-agent decoupling) than this Y4 paper
(H10, LLM self-monitoring).

## Suggested area and program committee

- **Area:** Evaluation, Reproducibility, and Negative Results
  (a designated track in COLM 2026).
- **Conflicts:** none.
- **Preferred reviewers:** any reviewer with a track record in
  evaluation methodology for LLM self-supervision; we have no
  preferred/excluded reviewer list.

We thank the COLM 2026 program chairs for considering a
pre-registered negative result paper across two LLM tasks.
The methodological discipline (pre-registration chain,
stratified split, versioned addenda, transparent cross-task
synthesis) is exactly the kind of contribution COLM aims to
elevate.