# Statement of Purpose -- Anthropic Research Scientist Application

> 刘泽文 (Liu Zewen)
> Target: Anthropic (research scientist, fall 2026)
> Archimedes Project (AGI-2026-001)
> Customized for Anthropic: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0
> Note: This is a research-scientist application, not a PhD application.
> The format is a research statement + extended-CV style document, not
> a US-style graduate SoP. Adjust tone accordingly.

---

## 1. Research Summary

I am applying to Anthropic as a research scientist to continue my
**Archimedes Project**, a 5-year independent research program toward a
self-improving AGI substrate. The central hypothesis is that **decoupling**
-- separating the failure-prediction Monitor from the policy gradient that
shapes behavior -- is the core mechanism enabling stable self-monitoring
in reinforcement-learning agents.

Across a pre-registered test programme (8 pre-registered H tests, 0
supported at the strict t>2 threshold), I have learned that the Monitor
is **informationally valid** (AUROC up to 0.99) but **not generically
useful as an online RL intervention**: 8/8 training-time and inference-
time uses of the Monitor failed to beat random-signal controls at the
pre-registered decision rule. The single validated use is **offline
verification and the evidence-chain governance substrate** (DLR 97.8%
mean across 4 envs; GovBench H1+H2 tampered detection 1.000; A2A trust
gate intercept_rate 1.000, n=7). The Y1 paper to be submitted (v3.7)
reports both the +50 LunarLander headline and the null cross-env / null
inference-time findings honestly.

The governance and evidence-chain research is the most relevant
contribution for an Anthropic application: it provides concrete,
tested primitives (PEP, A2A trust gate, tamper detection) that can be
deployed on top of real LLM agents.

## 2. Why Anthropic?

Anthropic is the right home for the second half of the Archimedes work
because of its unique focus on:

- **AI safety and alignment**: Anthropic's mission -- "AI that's
  helpful, honest, and harmless" -- aligns directly with the
  Archimedes governance substrate. The 8-test null synthesis on
  training-time Monitor interventions is also a safety-relevant
  result: it shows that *adding* auxiliary signals to an RL loop does
  not automatically make the agent safer, even when the signal is
  informative.
- **Mechanistic interpretability**: Anthropic's depth in interpreting
  the internal representations of large models is complementary to
  the Archimedes "verifier" framing. A decoupled Monitor is, in some
  sense, a *learned* interpretability signal: it gives a calibrated
  failure probability on a trajectory, which is the kind of thing
  that mechanistic interp work could dissect.
- **Scalable oversight and constitutional AI**: the Archimedes evidence-
  chain substrate (PEP + A2A trust gate + tamper detection) is a
  primitive that fits the constitutional-AI / scalable-oversight
  framework: it makes the agent's decision process auditable.
- **Honest empirical methodology**: Anthropic's tradition of rigorous
  experimental design, statistical testing, and null-result reporting
  (e.g. their well-known interpretability null results) aligns with
  the Archimedes pre-registered H framework.

I would like to spend my research-scientist years at Anthropic deepening
the Archimedes results in three specific ways:

1. **Deploying the evidence-chain substrate on real Claude-family
   agents**: extending GovBench H1+H2 (PEP violation_rate 0.000;
   tamper_detected 1.000; n=7) from deterministic-scripted agents to
   real LLM-based agents. The integration is straightforward but
   needs careful evaluation.
2. **Self-rewarding LLM agents**: extending the H1 decoupling result
   from PPO to RLHF and self-rewarding LLM agents. The frozen-Monitor
   logic should transfer directly, but no one has tested it at scale.
   This is directly relevant to scalable oversight.
3. **Mechanistic interpretation of the decoupled Monitor**: treating
   the Slot-Monitor (AUROC 0.989) as a learned interpretability
   signal, dissecting what it actually detects.

## 3. Background and Preparation

My preparation matches Anthropic's research focus:

- **AI safety and governance**: GovBench H1+H2+H3 validated (PEP
  violation_rate 0.000; tamper_detected 1.000; impersonation
  intercept_rate 1.000; n=7). Evidence-chain primitives are the
  Archimedes' most directly Anthropic-relevant contribution.
- **Self-monitoring and decoupling**: H1 ablation 5/5 seeds; pre-
  registered framework with 9 explicit hypotheses (6 validated, 2
  refuted, 1 open). The 8-test null synthesis is a safety-relevant
  finding.
- **Neuro-symbolic reasoning**: DLR attention architecture, 4-env
  validation (97.8% mean accuracy over 19 predicates). The DLR is a
  neuro-symbolic verification primitive.
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null. This matches Anthropic's null-result tradition.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. Anthropic's combination of safety mission,
interpretability depth, and constitutional-AI tradition is the right
environment to take the Archimedes governance substrate from
deterministic scripts to real LLM agents.

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
and public on GitHub. I will continue this practice in my research-
scientist work (with appropriate IP considerations for proprietary
research, which I am familiar with from the PUBLICATION_HOLD.md
precedent in the Archimedes repo).

## 5. Career Goals

After Anthropic, I plan to:

- Continue independent research in AGI safety and architecture.
- Publish in top-tier venues (NeurIPS, ICML, JMLR) where possible
  (consistent with Anthropic publication policies).
- Mentor junior researchers, especially those from underrepresented
  backgrounds.
- Maintain a public, reproducible research practice.

## 6. Why Me?

I bring to Anthropic:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **A pre-validated governance substrate** (GovBench H1+H2+H3) that
   can be deployed on real LLM agents with minimal engineering work.
3. **A pre-registered empirical framework** with hard decision rules
   (Welch t > 2.0 on pre-registered sample sizes), which is rare in
   independent RL research and aligns with Anthropic's empirical
   tradition.
4. **The H1 ablation** as a starting point for self-rewarding LLM
   agents (a natural Anthropic topic).
5. **The 8-test null synthesis** as a publishable safety-relevant
   negative-result contribution: *adding* auxiliary signals to an RL
   loop does not automatically make the agent safer.
6. **Open science practice** -- MIT license, public GitHub, public
   experiment logs (modulo Anthropic publication policy).
7. **Perseverance** -- 7+ years of independent study culminating in
   this work.

I am excited to bring this momentum to Anthropic.

---

*Word count: ~700. Adjust tone for research-scientist format; remove
"PhD years" references; this is a research statement, not a graduate
admissions SoP.*