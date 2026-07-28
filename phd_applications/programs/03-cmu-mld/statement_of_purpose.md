# Statement of Purpose -- CMU MLD PhD Application

> 刘泽文 (Liu Zewen)
> Target: CMU Machine Learning Department (MLD) / CSD
> Archimedes Project (AGI-2026-001)
> Customized for CMU MLD: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0

---

## 1. Research Summary

I am applying to CMU MLD to continue my **Archimedes Project**, a 5-year
independent research program toward a self-improving AGI substrate. The
central hypothesis is that **decoupling** -- separating the failure-
prediction Monitor from the policy gradient that shapes behavior -- is the
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

## 2. Why CMU MLD?

I am applying to CMU MLD because it is the deepest single place for
**decision-making under uncertainty, RL, and value-function methods**,
which are the exact intellectual neighbours of the Archimedes work:

- **RL foundations**: CMU's long-running work on policy-gradient
  methods, exploration, distributional RL, and value-function learning
  is the natural context for the H1 ablation and the Y1.3 follow-up.
- **Decision-making and planning**: the Archimedes work has shown that
  the Monitor signal is not useful as a direct RL intervention, which
  pushes the next research direction toward planning and explicit
  decision-making (model-based planning, hierarchical RL, options).
  This is exactly CMU's strength.
- **Multi-agent and self-play**: the Phase 2 multi-agent work
  (DMC, PettingZoo Simple Spread v3, MADDPG v2) is a Y2 direction;
  CMU's research depth in multi-agent RL, QMIX, MADDPG, and self-play
  is the right home for this.
- **Honest empirical methodology**: CMU's tradition of rigorous
  experimental design, statistical testing, and null-result reporting
  aligns naturally with the Archimedes pre-registered H framework.

I would like to spend my PhD years at CMU deepening the Archimedes
results in three specific ways:

1. **Decision-aware RL**: integrating the DLR (97.8% cross-env) with
   a planning loop on top of PPO, so that the symbolic predicate layer
   informs the policy's decision boundary rather than just acting as a
   verifier.
2. **Multi-agent Monitor coordination**: the H5 REFUTED result on
   PettingZoo shows that decoupled Monitors do not transfer to MA by
   default. The fix likely involves centralised critics (MADDPG v2
   baseline already shows +7.7 vs random, p<0.001) and shared predicate
   spaces, both of which are CMU's home turf.
3. **Honest cross-environment replication** of the Y1.3 Monitor-as-
   regularizer finding with multi-seed statistical power; the 8-test
   null synthesis is a publishable contribution in CMU's empirical-
   methodology tradition.

## 3. Background and Preparation

My preparation matches CMU MLD's research depth:

- **Reinforcement learning**: PPO, SAC, CQL, MADDPG v2, slot-attention
  world models; end-to-end pipeline on CPU, 100K-step budgets.
- **Self-monitoring and decoupling**: H1 ablation 5/5 seeds; pre-
  registered framework with 9 explicit hypotheses (6 validated, 2
  refuted, 1 open).
- **Multi-agent RL**: PettingZoo Simple Spread v3 baseline; per-agent
  Monitor (AUROC 0.99) ; DMC continuous-action 5-seed sweep; MADDPG v2
  baseline (the only working strong-positive Phase 2 baseline).
- **Neuro-symbolic reasoning**: DLR attention architecture, 4-env
  validation (97.8% mean accuracy over 19 predicates).
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. CMU's depth in decision-making, multi-agent RL, and
empirical methodology is the right environment to take Archimedes from
the single-agent RL baseline to multi-agent decision-aware policies.

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

I bring to CMU MLD:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **A pre-registered empirical framework** with hard decision rules
   (Welch t > 2.0 on pre-registered sample sizes), which is rare in
   independent RL research and aligns with CMU's empirical tradition.
3. **The H1 ablation** as a starting point for a CMU-level thesis on
   decoupled self-monitoring.
4. **The 8-test null synthesis** as a publishable negative-result
   contribution that grounds the next phase of work.
5. **A multi-agent Phase 2 baseline** (MADDPG v2 +7.7 vs random,
   p<0.001) ready to extend with the DLR predicate layer.
6. **Open science practice** -- MIT license, public GitHub, public
   experiment logs.
7. **Perseverance** -- 7+ years of independent study culminating in
   this work.

I am excited to bring this momentum to CMU MLD.

---

*Word count: ~700. Modify for specific lab / advisor once identified.*