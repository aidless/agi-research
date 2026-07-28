# Statement of Purpose -- DeepMind Research Scientist Application

> 刘泽文 (Liu Zewen)
> Target: DeepMind (research scientist, fall 2026)
> Archimedes Project (AGI-2026-001)
> Customized for DeepMind: 2026-07-28
> Source template: phd_applications/statement_of_purpose.md v1.0
> Note: This is a research-scientist application, not a PhD application.
> The format is a research statement + extended-CV style document, not
> a US-style graduate SoP. Adjust tone accordingly.

---

## 1. Research Summary

I am applying to DeepMind as a research scientist to continue my
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
mean across 4 envs; GovBench H1+H2 tampered detection 1.000). The Y1
paper to be submitted (v3.7) reports both the +50 LunarLander headline
and the null cross-env / null inference-time findings honestly.

I have also built a Phase 2 multi-agent baseline (PettingZoo Simple Spread
v3, MADDPG v2 +7.7 vs random, p<0.001) and documented that decoupled
per-agent Monitors do *not* transfer to multi-agent by default (H5
REFUTED on continuous actions, 1/5 positive seeds, t=-2.53). The
decentralized Monitors train to AUROC 0.99 in MA (decoupling holds),
but real Monitor shaping is worse than no shaping on continuous actions.

## 2. Why DeepMind?

DeepMind is the right home for the second half of the Archimedes work
because of its unique combination of:

- **Multi-agent RL and self-play**: the Phase 2 work (PettingZoo Simple
  Spread v3, MADDPG v2) is a Y2 direction; DeepMind's depth in
  multi-agent RL, self-play (AlphaStar, AlphaGo, AlphaTensor), and
  emergent communication is the natural place for this.
- **Self-improving agents and recursive self-modification**: DeepMind's
  history of building agents that improve themselves (AlphaZero,
  AlphaProof, AlphaEvolve) is the right intellectual neighbour of
  the Archimedes "self-improvement loop" (H9 OPEN).
- **Foundation-model + RL integration**: the Archimedes substrate
  needs to be interfaced with modern foundation models (the Project D
  "language as type system" direction); DeepMind's Gemini-family work
  and the Gemini-Robotics lineage is the natural context.
- **Honest empirical methodology**: DeepMind's tradition of large-scale
  empirical work (with proper pre-registration and null-result reporting
  in safety-relevant domains) aligns with the Archimedes pre-registered
  H framework.

I would like to spend my research-scientist years at DeepMind deepening
the Archimedes results in three specific ways:

1. **Multi-agent self-improvement**: extending the H5 REFUTED result
   with centralised critics (MADDPG v2 baseline already shows +7.7
   vs random, p<0.001) and shared predicate spaces, going beyond the
   simple "decoupled Monitor per agent" baseline that failed.
2. **Self-rewarding LLM agents**: extending the H1 decoupling result
   from PPO to RLHF and self-rewarding LLM agents. The frozen-Monitor
   logic should transfer directly, but no one has tested it at scale.
3. **Honest cross-environment replication** of the Y1.3 Monitor-as-
   regularizer finding with multi-seed statistical power; the 8-test
   null synthesis is a publishable contribution.

## 3. Background and Preparation

My preparation matches DeepMind's research breadth:

- **Reinforcement learning**: PPO, SAC, CQL, MADDPG v2, slot-attention
  world models; end-to-end pipeline on CPU, 100K-step budgets.
- **Self-monitoring and decoupling**: H1 ablation 5/5 seeds; pre-
  registered framework with 9 explicit hypotheses (6 validated, 2
  refuted, 1 open).
- **Multi-agent RL**: PettingZoo Simple Spread v3 baseline; per-agent
  Monitor (AUROC 0.99); DMC continuous-action 5-seed sweep; MADDPG v2
  baseline (the only working strong-positive Phase 2 baseline).
- **Neuro-symbolic reasoning**: DLR attention architecture, 4-env
  validation (97.8% mean accuracy over 19 predicates).
- **Pre-registration and honest reporting**: 8 pre-registered H tests
  with a pre-committed decision rule (Welch t > 2.0), even when the
  result is null.
- **Engineering**: 110+ commits on github.com/aidless/agi-research,
  MIT-licensed, full reproducibility on CPU.

I am self-taught in many areas; my independent work has been my primary
research vehicle. DeepMind's combination of multi-agent depth, self-
improving-agent tradition, and large-scale empirical methodology is
the right environment to take Archimedes from the single-agent RL
baseline to multi-agent, foundation-model-aware substrate.

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

After DeepMind, I plan to:

- Continue independent research in AGI safety and architecture.
- Publish in top-tier venues (NeurIPS, ICML, JMLR) where possible
  (consistent with DeepMind publication policies).
- Mentor junior researchers, especially those from underrepresented
  backgrounds.
- Maintain a public, reproducible research practice.

## 6. Why Me?

I bring to DeepMind:

1. **A working 5-year research program** with measurable milestones
   (110+ commits, 4 STRONG POSITIVES, 12+ honest negatives, 6 hypotheses
   validated, 1 OPEN).
2. **The H1 ablation** as a starting point for self-improving agents
   (a natural DeepMind topic).
3. **The 8-test null synthesis** as a publishable negative-result
   contribution that grounds the next phase of work.
4. **A multi-agent Phase 2 baseline** (MADDPG v2 +7.7 vs random,
   p<0.001) ready to extend with the DLR predicate layer.
5. **A governance substrate** (GovBench H1+H2+H3) that DeepMind's
   safety mission could integrate with Gemini-family agents.
6. **Open science practice** -- MIT license, public GitHub, public
   experiment logs (modulo DeepMind publication policy).
7. **Perseverance** -- 7+ years of independent study culminating in
   this work.

I am excited to bring this momentum to DeepMind.

---

*Word count: ~700. Adjust tone for research-scientist format; remove
"PhD years" references; this is a research statement, not a graduate
admissions SoP.*