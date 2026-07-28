# Statement of Purpose — MIT CSAIL PhD Application

> 刘泽文 (Liu Zewen)
> Target: MIT CSAIL, EECS / CSAIL AI/ML research groups
> Archimedes Project (AGI-2026-001)
> Customized for MIT CSAIL: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0

---

## 1. Research Summary

I am applying to your PhD program to continue my **Archimedes Project**, a
5-year independent research program toward a self-improving AGI substrate.
The central hypothesis is that **decoupling** — separating the failure-
prediction Monitor from the policy gradient that shapes behavior — is the
core mechanism enabling stable self-monitoring in reinforcement-learning
agents.

The Archimedes Project is grounded in the ENWI framework, a 5-layer
architecture with 11 mathematical theorems and five falsifiable predictions.
I have ported four ENWI components into a working codebase (Active Inference
Engine, Differentiable Logic Reasoner, Composable Physics, Slot Attention)
and validated the central H1 ablation across 5 random seeds on
LunarLander-v3 (delta = 0.724, AUROC frozen vs joint Monitor, 5/5 seeds
support decoupling).

Across a broader pre-registered test programme (8 pre-registered H tests,
0 supported at the strict t>2 threshold), I have learned that the Monitor
is **informationally valid** (AUROC up to 0.99) but **not generically
useful as an online RL intervention**: 8/8 training-time and inference-
time uses of the Monitor failed to beat random-signal controls at the
pre-registered decision rule. The single validated use is **offline
verification and the evidence-chain governance substrate** (DLR 97.8%
mean across 4 envs; GovBench H1+H2 tampered detection 1.000). The paper
to be submitted (v3.7) reports both the +50 LunarLander headline and the
null cross-env / null inference-time findings honestly.

## 2. Why MIT CSAIL?

I am applying to MIT CSAIL because of the depth and breadth of its research
in **reinforcement learning, robotics, and self-improving systems**.
CSAIL groups have published directly on the problems I want to spend my
PhD years deepening:

- **Decoupled critics and policy improvement**: CSAIL's long-running
  work on model-based RL, offline RL (CQL, IQL), and value-function
  methods is the natural intellectual context for the H1 ablation.
- **Self-improving agents and meta-learning**: the intersection of
  meta-learning, RL, and LLM-based agents is exactly the territory I
  want to keep probing beyond the 8-test null result.
- **Symbolic and neuro-symbolic verification**: the program-wide
  strength in formal methods, program synthesis, and differentiable
  logic aligns with the DLR (Differentiable Logic Reasoner) component
  of the Archimedes substrate.

I would like to spend my PhD years at CSAIL deepening the Archimedes
results, specifically:

1. **Cross-environment replication** of the Y1.3 Monitor-as-regularizer
   finding with multi-seed statistical power (the current n=15 result is
   single-env; the 8-test null result is multi-env but small-N).
2. **Real self-improvement loops**: H9 (the only OPEN hypothesis in my
   framework) requires multi-step Monitor -> PPO -> new Monitor -> new PPO
   cycles, which is exactly the kind of empirical systems work CSAIL
   excels at.
3. **Differentiable logic as a verification primitive**: integrating
   the DLR (97.8% cross-env accuracy) with a real policy loop,
   particularly in safety-critical RL settings.

## 3. Background and Preparation

My preparation matches CSAIL's research depth:

- **Reinforcement learning**: PPO, SAC, CQL, slot-attention world models;
  end-to-end pipeline on CPU, 100K-step budgets.
- **Self-monitoring and decoupling**: H1 ablation 5/5 seeds; pre-registered
  framework with 9 explicit hypotheses (6 validated, 2 refuted, 1 open).
- **Neuro-symbolic reasoning**: DLR attention architecture, 4-env
  validation (97.8% mean accuracy over 19 predicates), slot attention
  adapted to 1-D trajectory sequences.
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null. This methodology is publishable in itself.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. CSAIL's combination of depth, mentorship, peer
network, and resources is what I need to take the Archimedes results
from "single-agent RL on LunarLander" to "cross-domain, multi-agent,
peer-reviewed substrate".

## 4. Research Philosophy

I work in **AIKR mode** (Assumption of Insufficient Knowledge and
Resources, after Pei Wang's NARS): acknowledge uncertainty, iterate
under bounded compute, report negative results with the same precision
as positive ones. The Archimedes Project documents this commitment:
the thesis and Y1 paper include both the +50 LunarLander result and
the 8-test null outcome, the DEC-0011 HALT after 6 inference-time
failures, the ENWI Prediction-2 non-replication, and the self-correction
sequence (v1.0 -> v1.1 -> v1.2 -> v1.3 -> v1.4) after the Y1.3 overclaim.

I also work in **open science**: all Archimedes code is MIT-licensed
and public on GitHub. I will continue this practice in my PhD work.

## 5. Career Goals

After PhD, I plan to:

- Continue independent research in AGI safety and architecture.
- Publish in top-tier venues (NeurIPS, ICML, JMLR).
- Mentor junior researchers, especially those from underrepresented
  backgrounds.
- Maintain a public, reproducible research practice.

## 6. Why Me?

I bring to CSAIL:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **Pre-registered empirical methodology** with hard decision rules
   (Welch t > 2.0 on pre-registered sample sizes), which is rare in
   independent RL research.
3. **The H1 ablation** as a concrete starting point for a CSAIL-level
   thesis on self-monitoring.
4. **The 8-test null synthesis** as a publishable negative-result
   contribution that saves the next experimenter from redoing the work.
5. **Open science practice** -- MIT license, public GitHub, public
   experiment logs.
6. **Perseverance** -- 7+ years of independent study culminating in this
   work.

I am excited to bring this momentum to MIT CSAIL.

---

*Word count: ~700. Modify for specific lab / advisor once identified.*