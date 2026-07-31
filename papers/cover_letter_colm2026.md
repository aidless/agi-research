# Cover Letter -- COLM 2026 Submission

**To:** COLM 2026 Program Chairs
**From:** Liu Zewen, Archimedes Project (AGI-2026-001)
**Date:** July 31, 2026
**Re:** Submission of "When Decoupling Does Not Help LLM
Self-Monitoring Either: A Pre-Registered n=5/20/100 Replication"

---

Dear Program Chairs,

We are pleased to submit our paper "When Decoupling Does Not
Help LLM Self-Monitoring Either: A Pre-Registered n=5/20/100
Replication" for consideration at COLM 2026.

## Why this paper is a good fit for COLM

COLM welcomes rigorous empirical studies, including negative
results that reframe the limits of an existing technique. This
paper is exactly that kind of contribution: a pre-registered
test of whether the failure-prediction Monitor architecture --
verified in single-agent RL (Y1 paper, n=15 seeds, t=6.76,
p<0.001) and shown to fail in multi-agent RL (Y3 paper, 5/6
pathways REFUTED at p<0.05) -- transfers to LLM self-monitoring
in the Qwen2.5-1.5B setting.

The paper's central finding: H10 (decoupling transfers to LLM
self-monitoring) is REFUTED at all three sample sizes we tested
(n=5, n=20, n=100). The 95% bootstrap CI for the Frozen - Joint
contrast is [-0.087, +0.117] at n=100; all three arms (Frozen,
Joint, Random) are within +/- 0.02 of 0.5 (random). The effect
sizes are too small to be practically meaningful, and the
direction is not stable across replications (n=5 shows Joint >
Frozen, n=20 and n=100 show Frozen > Joint, all consistent with
sampling noise on a near-zero effect).

## Why a pre-registered negative result matters for COLM

The LLM community has been increasingly interested in self-
monitoring (e.g., self-consistency, selective prediction,
calibration-based methods, runtime guardrails). This paper
provides empirical evidence that one specific architectural
claim (decoupled Monitor) does NOT transfer to LLM self-
monitoring on simple arithmetic tasks. By publishing the full
pre-registered protocol (H10 decision rule, sample sizes,
analysis pipeline) and the three independent replications, we
enable the community to know exactly what was tried, what was
not, and what the next-step hypothesis should be (a harder
trace task, e.g., GSM8K 200+ token rollouts).

The paper also contributes a methodological lesson: the
stratified train/eval split (Section 3 of the paper) is a
better default for self-monitoring evaluation than the
deterministic split used in earlier work, since the
deterministic split can collapse to a single class on some
seeds and make AUROC undefined. The rebalance fallback we
introduce is a small, reusable piece of code that we hope other
COLM submissions will adopt.

## Compute and reproducibility

The n=100 H10 pilot took 8h51m wall-clock on CPU (Qwen2.5-1.5B
Instruct, 300 jobs at MAX_PARALLEL=1). All code, logs, and
aggregated JSON data are committed to the Archimedes Project
git repository (`experiments_log/_h10_n100_*.log`,
`experiments_log/_h10_n100_bootstrap.json`,
`experiments_log/_h10_n100_forest.png`). A supplementary
materials document (S1-S12) provides full provenance for
every number in the paper.

## What this paper does not claim

We do not claim that the failure-prediction Monitor can never
help with LLM self-monitoring -- only that the simple
arithmetic trace at the 1.5B parameter scale does not produce a
strong enough signal for the Monitor architecture to learn
from, regardless of how the Monitor is trained (frozen or
joint). The 95% bootstrap CI is consistent with both a true
zero effect and a small positive effect of up to 0.12 AUROC.
The honest conclusion is that the simple arithmetic task is
not the right test bed for H10, and a future GSM8K 200+ token
pilot is the natural next step (not within the scope of this
submission).

## Conflict of interest and prior publication

This paper is original work. The Monitor architecture itself is
prior work (Y1 paper, Y3 paper, Y4 v0.4) but the n=20 and n=100
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
pre-registered negative result paper. We believe the
methodological discipline (pre-registration, three sample
sizes, transparent reporting of effect-shrinkage) is exactly
the kind of contribution COLM aims to elevate.
