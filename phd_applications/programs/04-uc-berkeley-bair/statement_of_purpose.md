# Statement of Purpose -- UC Berkeley BAIR PhD Application

> 刘泽文 (Liu Zewen)
> Target: UC Berkeley BAIR / EECS
> Archimedes Project (AGI-2026-001)
> Customized for UC Berkeley BAIR: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0

---

## 1. Research Summary

I am applying to UC Berkeley BAIR to continue my **Archimedes Project**, a
5-year independent research program toward a self-improving AGI substrate.
The central hypothesis is that **decoupling** -- separating the failure-
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

## 2. Why UC Berkeley BAIR?

I am applying to UC Berkeley BAIR because it is the natural home for
**slot-attention world models and object-centric RL**, which is the
exact methodological backbone of the Archimedes work:

- **Slot attention is a Berkeley contribution** (Locatello et al.
  NeurIPS 2020), and BAIR continues to lead on object-centric scene
  decomposition, which is what my slot-attention world model uses.
  The intellectual context for extending slot attention to 1-D
  trajectory sequences (the Archimedes slot world model, err 0.000007)
  is strongest here.
- **World models and planning**: BAIR's strength in world models
  (Dreamer lineage, JEPA-related work, decision-aware world models)
  is exactly what the Archimedes substrate needs next. The H1 ablation
  result and the Y1.3 follow-up both involve world-model-adjacent
  signals; BAIR is the place to push this further.
- **RL theory and exploration**: the Archimedes work has documented
  that the Monitor is a useful *verifier*, not a useful *reward signal*
  (8/8 training-time and inference-time uses failed). The natural next
  direction is to use the Monitor as a verifier inside a world-model
  planning loop; BAIR's tradition of clean theoretical framing fits
  this.
- **Cross-method integration**: BAIR's openness to combining deep
  learning, classical RL, control theory, and probabilistic inference
  matches the ENWI framework's hybrid style.

I would like to spend my PhD years at Berkeley deepening the Archimedes
results in three specific ways:

1. **Slot-attention world-model scaling**: extending the 1-D trajectory
   slot world model to harder envs (Atari, Procgen), with learned
   (not hand-coded) predicates and OOD generalization tests.
2. **Verifier-aware planning**: integrating the DLR (97.8% cross-env)
   as a verifier inside a world-model planning loop, so that the
   policy can plan *through* the symbolic predicate layer.
3. **Honest cross-environment replication** of the Y1.3 result with
   multi-seed statistical power; the 8-test null synthesis is a
   publishable contribution in BAIR's empirical tradition.

## 3. Background and Preparation

My preparation matches BAIR's research depth:

- **Slot attention and object-centric models**: Slot-Monitor +0.193
  AUROC vs raw-history Monitor; slot world model 1-D adaptation;
  DLR attention architecture (97.8% cross-env accuracy).
- **World models**: slot_dynamics.py reconstruction error 0.000007;
  end-to-end PPO pipeline on CPU, 100K-step budgets.
- **Reinforcement learning**: PPO, SAC, CQL, slot-attention world models,
  model-based planning baselines (MBP slot+DLR, NEGATIVE result
  documented honestly).
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. BAIR's combination of slot-attention depth, world-
model tradition, and clean empirical methodology is the right
environment to take Archimedes from the slot world model baseline to
scaled, verifier-aware planning.

## 4. Research Philosophy

I work in **AIKR mode** (Assumption of Insufficient Knowledge and
Resources, after Pei Wang's NARS): acknowledge uncertainty, iterate
under bounded compute, report negative results with the same precision
as positive ones. The Archimedes Project documents this commitment:
the thesis and Y1 paper include both the +50 LunarLander result and
the 8-test null outcome, the DEC-0011 HALT after 6 inference-time
failures, the ENWI Prediction-2 non-replication, the MBP -273 result,
and the self-correction sequence (v1.0 -> v1.1 -> v1.2 -> v1.3 -> v1.4)
after the Y1.3 overclaim.

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

I bring to BAIR:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **Direct slot-attention lineage work**: the Archimedes Slot-Monitor
   and slot world model extend Locatello 2020's slot attention to
   trajectory sequences and failure-prediction, with strong empirical
   evidence (Slot-Monitor AUROC 0.989, slot world model err 0.000007).
3. **The H1 ablation** as a starting point for a BAIR-level thesis on
   decoupled self-monitoring.
4. **The 8-test null synthesis** as a publishable negative-result
   contribution that grounds the next phase of work.
5. **Open science practice** -- MIT license, public GitHub, public
   experiment logs.
6. **Perseverance** -- 7+ years of independent study culminating in
   this work.

I am excited to bring this momentum to UC Berkeley BAIR.

---

*Word count: ~700. Modify for specific lab / advisor once identified.*