# Statement of Purpose -- Stanford PhD Application

> 刘泽文 (Liu Zewen)
> Target: Stanford CS / Stanford AI Lab (SAIL) / HAI
> Archimedes Project (AGI-2026-001)
> Customized for Stanford: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0

---

## 1. Research Summary

I am applying to Stanford CS / SAIL / HAI to continue my **Archimedes
Project**, a 5-year independent research program toward a self-improving
AGI substrate. The central hypothesis is that **decoupling** -- separating
the failure-prediction Monitor from the policy gradient that shapes
behavior -- is the core mechanism enabling stable self-monitoring in
reinforcement-learning agents.

The Archimedes Project is grounded in the ENWI framework, a 5-layer
architecture with 11 mathematical theorems and five falsifiable predictions.
I have ported four ENWI components into a working codebase (Active Inference
Engine, Differentiable Logic Reasoner, Composable Physics, Slot Attention)
and validated the central H1 ablation across 5 random seeds on
LunarLander-v3 (delta = 0.724, AUROC frozen vs joint Monitor, 5/5 seeds
support decoupling).

Across a broader pre-registered test programme (8 pre-registered H tests,
0 supported at the strict t>2 threshold), the Archimedes work has produced
both a substantial positive result (Y1.3, +50 LunarLander, n=15, p<0.001)
and a comprehensive null synthesis (8 pre-reg tests, 0 supported). The
single validated use of the Monitor is **offline verification and the
evidence-chain governance substrate** (DLR 97.8% mean across 4 envs;
GovBench H1+H2 tampered detection 1.000). The paper to be submitted
(v3.7) reports both the headline result and the null synthesis honestly.

## 2. Why Stanford CS / SAIL / HAI?

Stanford is the right home for the second half of the Archimedes work
because of its unique combination of:

- **Foundation-model fluency** at SAIL and HAI -- the Archimedes
  substrate needs to be interfaced with modern foundation models (the
  Project D "language as type system" direction), and Stanford's depth
  in language, vision, and multi-modal foundation models is the
  natural context for this.
- **Long-running RL and self-improvement tradition** -- Stanford's work
  on RLHF, instruction tuning, and self-rewarding agents is the
  intellectual neighbour of my H1 ablation and Y1.3 result. The same
  decoupling logic that makes Monitor-vs-PPO separation valuable in
  classical RL likely also applies to LLM self-rewarding.
- **AI safety and governance** as a research direction -- HAI's
  explicit mission on human-centered AI aligns with the Archimedes
  evidence-chain / governance substrate (GovBench H1-H8 validations).
  This is where I want to spend my time after the single-agent RL
  baselines are done.
- **Cross-disciplinary breadth** -- the Archimedes substrate touches
  RL, vision, language, formal methods, and governance; Stanford's
  culture of cross-department collaboration is exactly the environment
  this needs.

I would like to spend my PhD years at Stanford deepening the Archimedes
results in three specific ways:

1. **Self-rewarding LLM agents**: extending the H1 decoupling result
   from PPO to RLHF and self-rewarding LLM agents. The frozen-Monitor
   logic should transfer directly, but no one has tested it.
2. **Foundation-model interfaces** to the slot-attention world model
   and the DLR (Differentiable Logic Reasoner). I want to use a small
   LM (Qwen-1.5B-class) as the Project D type-system layer.
3. **Governance and evidence-chain research**: integrating the
   Archimedes evidence-chain primitives (PEP, A2A trust gate,
   tamper-detection) with real LLM-based agents, going beyond the
   deterministic-scripted GovBench baseline.

## 3. Background and Preparation

My preparation matches Stanford's research breadth:

- **Reinforcement learning**: PPO, SAC, CQL, slot-attention world models;
  end-to-end pipeline on CPU, 100K-step budgets.
- **Self-monitoring and decoupling**: H1 ablation 5/5 seeds; pre-registered
  framework with 9 explicit hypotheses (6 validated, 2 refuted, 1 open).
- **Neuro-symbolic reasoning**: DLR attention architecture, 4-env
  validation (97.8% mean accuracy over 19 predicates).
- **Foundation-model-adjacent infrastructure**: slot attention adapted
  to 1-D trajectory sequences; LLM-as-type-system templating (Project D).
- **Governance and evidence chain**: GovBench H1+H2+H3 validated
  (PEP violation_rate 0.000; tamper_detected 1.000; impersonation
  intercept_rate 1.000; n=7).
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. Stanford's combination of foundation-model depth, RL
tradition, and AI-safety mission is the right environment to take
Archimedes from single-agent baselines to foundation-model-aware
substrate.

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

I bring to Stanford:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **The H1 ablation** as a starting point for self-rewarding LLM
   agents (a natural Stanford topic).
3. **The 8-test null synthesis** as a publishable negative-result
   contribution that grounds the next phase of work.
4. **A governance substrate** (GovBench H1+H2+H3) that Stanford's HAI
   mission could integrate with real LLM agents.
5. **Open science practice** -- MIT license, public GitHub, public
   experiment logs.
6. **Perseverance** -- 7+ years of independent study culminating in this
   work.

I am excited to bring this momentum to Stanford.

---

*Word count: ~700. Modify for specific lab / advisor once identified.*