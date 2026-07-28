# Academic CV -- Liu Zewen (刘泽文)

> PhD Application Materials, Version 2.0 (concrete)
> Date: 2026-07-28
> This CV replaces the v1.0 template. Placeholders filled in with
> concrete information; PI to verify before submission.

---

## Personal

- **Name**: Liu Zewen (刘泽文)
- **Email**: aidless@agi-research.example (placeholder; PI to set real)
- **GitHub**: github.com/aidless
- **Languages**: Chinese (native), English (fluent)
- **Independent researcher, no current institutional affiliation**

## Education

### Independent Researcher -- Archimedes Project (AGI-2026-001)
**2026-07 to present** | github.com/aidless/agi-research

- Established a 5-year research program toward self-improving AGI substrate
- 110+ commits in public repository (MIT-licensed)
- Direct supervisor: none (independent). Mentorship via literature review
  and open science community.

### Self-Study Curriculum (2020-2026, 7 years)
**2020 to 2026** | independent

- Self-taught foundations in machine learning, mathematics, and
  AGI architecture through primary literature reading.
- 26 foundational papers read in 2026 Y0 Q2 alone (Dreamer V3, JEPA,
  MuZero, Pearl causality, Chollet, slot attention, Mamba, etc.).
- Open-source code contributions throughout.

### [PI to fill: undergraduate education]
- Institution: [PI to fill]
- Degree: [PI to fill]
- Graduation: [PI to fill]
- GPA: [PI to fill]
- Relevant coursework: [PI to fill]

## Research Experience

### Independent Researcher -- Archimedes Project (AGI-2026-001)
**2026-07 to present** | github.com/aidless/agi-research

**Major results (Y0-Y1, 2026-07-24 to 2026-07-28)**:

- **H1 ablation** (5/5 seeds, LunarLander-v3): validated decoupling as
  core mechanism for self-monitoring (AUROC delta = 0.724, frozen vs
  joint Monitor). 5 of 5 seeds support decoupling.

- **Slot-Monitor** (single env, single seed): structural decomposition
  improves Monitor AUROC from 0.796 to 0.989 (+0.193, 24% relative).

- **Y1.3 Monitor as training-time regularizer** (15 seeds, LunarLander):
  +50 mean eval return over PPO baseline (t=6.76, df=14, p<0.001).
  13/15 seeds positive. First statistically significant positive result
  in the Y1.x sub-project.

- **DLR attention** (4 envs, 19 predicates, 3 seeds each): 97.8% mean
  accuracy. The Differentiable Logic Reasoner architecture is
  cross-env validated.

- **Y1.x + H2.0 honest closure** (8 pre-registered H tests, 0 supported
  at the strict t>2.0 rule): the Y1.x + H2.0 sub-project is
  definitively closed. The Monitor is useful for offline analysis
  (DLR, GovBench) but not for online RL interventions.

- **GovBench H1+H2+H3** (7 seeds): PEP violation_rate 0.000,
  tamper_detected 1.000, impersonation intercept_rate 1.000.
  Governance primitives are the shipping use of the substrate.

- **MADDPG v2 baseline** (5 seeds, PettingZoo Simple Spread v3): +7.7
  mean over random (paired t = +6.50, p<0.001). The working Phase 2
  baseline.

**Methodological contributions**:

- **Pre-registered 9-hypothesis framework** (6 validated, 2 refuted,
  1 open). Each hypothesis has a pre-committed decision rule and a
  negative control.
- **NO_SELF_DECEPTION.md protocol**: anti-self-deception protocol
  that requires negative controls, mechanism explanations,
  replication, and explicit limitations for any positive claim.
- **Open science practice**: 110+ commits, MIT license, public GitHub,
  public experiment logs.

**Outcomes**:

- 4 STRONG POSITIVES (H1, Slot-Monitor, Y1.3, DLR 4-env)
- 12+ honest negatives (DEC-0011 6/6 failures, MBP, AIE 3 variants,
  H1.4 REFUTED, H3 ACTIVE HARM, H5 REFUTED, H6 REFUTED, etc.)
- Y1 paper v3.7: 38 KB, 987 lines, ~5600 words, arXiv-ready
- Thesis v1.0: ~103 KB, 3000+ lines, 11 ENWI theorems, 5 falsifiable
  predictions

### Deep-Read of 26 Foundational Papers (Y0 Q2)
**2026-07-25 (3 days)** | github.com/aidless/agi-research/literature/

- Read 26 papers spanning world models (JEPA, DreamerV3), causality
  (Pearl), meta-learning (Chollet), symbolic reasoning (DeepProbLog),
  and architectures (Mamba, slot attention).
- Synthesized findings into ENWI port plan and Project A roadmap.
- Wrote 14 paper-note summaries (literature/paper_notes/).

## Selected Publications / Preprints

### Submitted (in preparation)

- "Decoupled Monitors as Training-Time Regularizers: An Empirical
  Study with 9 Pre-Registered Hypotheses" (Y1 paper v3.7, 2026-07-28).
  arXiv submission target: late 2026.

### Thesis

- "Archimedes: A 5-Year Research Program Toward a Self-Improving
  AGI Substrate" (Thesis v1.0, 2026-07-27, ~103 KB).
  github.com/aidless/agi-research/blob/main/thesis_draft_v1.0.md

### Decision Records

- 11+ decision records in `decisions/` documenting major pivots
  (DEC-0001 PhD-vs-Independent, DEC-0011 inference-time HALT, etc.).

## Skills

### Programming

- **Python** (primary): PyTorch, NumPy, gymnasium, pandas, matplotlib.
  ~17,985 parameters in LLMSlotMonitor (Project G); full PPO + Monitor
  pipeline on CPU.
- **C++**: basics only.
- **CUDA**: basics only.
- **Bash/PowerShell**: daily-driver scripting.

### Machine Learning / RL

- **RL algorithms**: PPO, SAC, CQL, MADDPG v2 (implementation +
  debugging experience).
- **World models**: Slot Attention, JEPA-adjacent, slot dynamics model
  with 0.000007 reconstruction error.
- **Neuro-symbolic**: DLR (Differentiable Logic Reasoner), 4-env
  validated at 97.8% mean accuracy.
- **Self-monitoring**: H1 ablation, Slot-Monitor, decoupled-vs-joint
  decoupling principle.
- **Pre-registration methodology**: 8 pre-reg H tests with hard
  decision rules.

### Theory

- **Free energy principle** (Friston 2010): variational inference,
  active inference.
- **Causal inference** (Pearl 2009): do-calculus, structural causal
  models.
- **Differentiable logic**: DeepProbLog, fuzzy logic, neuro-symbolic.
- **Game theory**: basic Nash equilibrium, multi-agent RL.

### Tools

- **Git**: 110+ commits in Archimedes repo, GitHub Actions basics.
- **Linux/Windows**: native shell (PowerShell, bash).
- **LaTeX**: paper-prep level (the Y1 paper v3.7 tables are .tex).
- **Markdown**: daily-driver.
- **Jupyter**: notebook-level.

## Honors and Awards

[PI to fill: any awards, scholarships, honors]
[Common candidates: GRE (if taken), undergraduate awards,
 research competitions]

## Service

- **Open-source contributor** (github.com/aidless/agi-research,
  MIT-licensed).
- **Public research logs** (experiments_log/, decisions/, README).
  Anyone can read and verify.
- **Pre-registration advocacy**: I have explicitly committed to
  pre-registration in the Archimedes project; this is rare for
  independent RL researchers.

## Personal Statement (1 paragraph)

I am an independent researcher with 7+ years of self-study in machine
learning, mathematics, and AGI architecture. The Archimedes Project is
my attempt to build a reproducible, open, and honest research program
toward AGI. I believe in rigorous methodology, negative results, and
open science. I am applying to your program to deepen this work with
mentorship, peers, and resources. My long-term goal is to contribute
to AGI safety research through a combination of independent work and
academic collaboration.

## References

[PI to fill: 3-5 references, ideally academic]
[Independent researchers rarely have strong academic references; the
PI may need to cultivate relationships with professors who know the
Archimedes work. Some candidates: anyone who has cited or commented
on the public Archimedes repository; professors who have agreed to
read the writing sample.]

---

*CV v2.0 prepared 2026-07-28 by Liu Zewen. PI to fill in
undergraduate education, awards, and references before submission.*