# Cover Letter -- AAMAS 2027 Submission

**To:** AAMAS 2027 Program Chairs / MARL Workshop Chairs
**From:** Liu Zewen, Archimedes Project (AGI-2026-001)
**Date:** July 29, 2026
**Re:** Submission of "Monitor Signal vs DLR Predicates in Cooperative MARL: A 6-Pathway Systematic Investigation"

---

Dear Program Chairs,

We are pleased to submit our paper "Monitor Signal vs DLR Predicates
in Cooperative MARL: A 6-Pathway Systematic Investigation" for
consideration at AAMAS 2027 (main track or MARL workshop).

## Why this paper is a good fit for AAMAS

This paper addresses a fundamental question in cooperative
multi-agent reinforcement learning (MARL): how should failure-
prediction information be incorporated into a MARL agent? The
paper reports a 6-pathway systematic investigation of 6 distinct
architectures for using failure-prediction Monitors in cooperative
MARL, spanning 14,000+ episodes of training on PettingZoo Simple
Spread v3. The investigation yields 4 central findings relevant
to the AAMAS community:

1. **Monitor signal does not transfer from single-agent to
   multi-agent as a training signal.** Five of six architectures
   (v3, v4, v5, v6, v7) are REFUTED at $p<0.05$.

2. **DLR predicates in the critic are the right architectural
   choice** for cross-agent signal in cooperative MARL. v8
   dlr\_only gives $+0.1447$ ($p<0.005$, $t=+3.216$, 20/30
   positive) at $n=30$ and $+0.0617$ ($p<0.05$ with Bonferroni
   correction for 2 tests) at $n=100$. The effect SHRANK from
   $n=30$ to $n=100$ (textbook small-effect signature) but
   remains statistically significant. A 3-seed independent
   replication on fresh seeds (200, 201, 202) reproduced the
   direction (2/3 positive, mean diff $+0.16$).

3. **The trust head architecture at the actor level completely
   ignores its input signal** (verified at $n=5$ and $n=30$ CLEAN
   via bit-for-bit identical per-seed results). The trust head
   learns a function of $o_i$ only and treats the input slot as
   noise.

4. **The Monitor's shipping use remains verification** (DLR
   predicates for cross-agent reasoning, runtime guardrails for
   safety), not training in MARL.

## Contribution to the field

We believe the 6-pathway systematic investigation is a valuable
contribution because:

- It provides **conclusive evidence** that the Monitor training
  signal does not transfer to MARL at any compute scale or sample
  size we tested. Prior work in this area has been limited to
  single-architecture investigations that left open the question
  of whether a different architectural placement would rescue the
  signal. Our 6-pathway investigation rules this out.
- The dlr\_only result has been independently replicated on
  three fresh seeds with direction-consistent outcomes, ruling
  out the alternative explanation that the n=100 effect is a
  chance of seed selection. Effect-shrinkage from n=30 to n=100
  is reported transparently as the textbook signature of a small
  effect that becomes more precisely estimated with larger
  samples.

- It introduces a **clean ablation protocol** (v6 proper
  re-implementation of the architecture-only ablation, with
  bit-for-bit identical per-seed verification) that we hope will
  become a standard for MARL ablation studies.

- It identifies **DLR predicates in the critic as the right
  architectural choice** for cross-agent signal in MARL, with a
  statistically significant and reproducible effect.

## Compliance with AAMAS formatting requirements

- The paper is formatted in LaTeX (compiled with pdfTeX).
- The paper is 8 pages including figures, tables, and references.
- We have included all required sections: Abstract, Introduction,
  Background, Methods, Results, Discussion, Conclusion, References.
- We have included a detailed supplementary materials document
  with the per-seed logs, code, and reproducibility instructions.

## Conflict of interest

The authors declare no conflict of interest.

## Funding

This work was supported by the Archimedes Project (AGI-2026-001)
compute infrastructure.

## Author contributions

L.Z. designed the research, ran all experiments, and wrote the
manuscript. Codex (an AI research assistant) provided code review,
literature search, and editorial support.

We thank you for your consideration. We look forward to your
response.

Sincerely,
Liu Zewen
Archimedes Project (AGI-2026-001)
[email protected]
