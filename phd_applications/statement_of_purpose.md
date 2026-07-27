# Statement of Purpose — 刘泽文 (Liu Zewen)

> PhD Application Materials, Version 1.0
> Date: 2026-07-27
> Project: Archimedes (AGI-2026-001)

---

## 1. Research Summary

I am applying to your PhD program to continue my **Archimedes Project**,
a 5-year independent research program toward a self-improving AGI substrate.
The central hypothesis is that **decoupling** — separating the failure-prediction
Monitor from the policy gradient that shapes behavior — is the core mechanism
enabling stable self-monitoring in reinforcement-learning agents.

The Archimedes Project is grounded in the ENWI (Embodied Neurosymbolic
World-model Intelligence) framework, a 5-layer architecture with 11 mathematical
theorems and five falsifiable predictions. I have ported four ENWI components
into a working codebase (Active Inference Engine, Differentiable Logic Reasoner,
Composable Physics, Slot Attention) and validated the central H1 ablation
across 5 random seeds on LunarLander-v3 (delta = 0.724, AUROC frozen vs joint
Monitor, 5/5 seeds support decoupling).

I have also documented honest negative results: ENWI''s central Prediction 2
(composable physics outperforms monolithic by 94%) does not replicate at our
scale; inference-time action gating on LunarLander fails 6/6 experiments;
Active Inference does not converge at our compute budget. These negative
results are reported with the same precision as positive ones.

## 2. Why This Program?

I am interested in your program because of [specific lab, advisor, or research
group]. My Archimedes Project aligns with [specific research area] at your
institution. I am particularly excited by [recent paper or research direction
from your group], which connects directly to my work on [specific topic].

The ENWI framework''s emphasis on **composability**, **symbolic verification**,
and **active inference** matches your group''s research focus on [match].
I would like to spend my PhD years deepening the Archimedes results, particularly:

1. **Multi-seed verification of H1** at 10+ seeds across multiple environments.
2. **Cross-environment transfer** of slot-attention world models.
3. **Real self-improvement loops** using Monitor feedback to update PPO (the
   Y1.3 direction, which produced the first POSITIVE inference result).
4. **Submission** of a comprehensive thesis to a top-tier conference
   (NeurIPS / ICML) by Y2 Q3.

## 3. Background and Preparation

I have a strong background in:

- **Reinforcement learning**: PPO, SAC, CQL, slot attention world models
- **Deep learning**: PyTorch, custom architectures (slot attention, GRU-AIE)
- **Mathematics**: free energy principle, variational inference, differentiable
  logic, control theory
- **Software engineering**: 86+ commits in the public Archimedes repo
  (github.com/aidless/agi-research), full reproducibility, MIT license

I am self-taught in many areas; my independent work has been my primary
research vehicle. Your program will give me the mentorship, peer network,
and resources to deepen this work.

## 4. Research Philosophy

I believe in **AIKR mode** (Assumption of Insufficient Knowledge and
Resources, after Pei Wang''s NARS): acknowledge uncertainty, iterate under
bounded compute, report negative results with the same precision as positive
ones. The Archimedes Project documents this commitment: the thesis includes
both successes (H1 5/5) and honest failures (DEC-0011 HALT, ENWI P2
non-replication).

I also believe in **open science**: all Archimedes code is MIT-licensed and
public on GitHub. I will continue this practice in my PhD work.

## 5. Career Goals

After PhD, I plan to:
- Continue independent research in AGI safety and architecture.
- Publish in top-tier venues (NeurIPS, ICML, JMLR).
- Mentor junior researchers, especially those from underrepresented
  backgrounds.
- Maintain a public, reproducible research practice.

## 6. Why Me?

I bring to your program:

1. **A working 5-year research program** with measurable milestones.
2. **75+ commits** of code, fully reproducible on CPU.
3. **Honest reporting** of both positive and negative results.
4. **Strong theoretical foundation** in ENWI, JEPA, free energy, slot attention.
5. **Open science practice** — MIT license, public GitHub.
6. **Perseverance** — 7+ years of independent study culminating in this work.

I am excited to bring this momentum to your program.

---

*Word count: ~600 (target 500-1000). Modify for specific program.*

## References

1. ENWI Paper (2026). F:\TMLR\Fusion\ENWI_PAPER.md.
2. Locatello et al. (2020). Object-Centric Learning with Slot Attention. NeurIPS.
3. Friston (2010). The Free-Energy Principle. Nature Reviews Neuroscience.
4. LeCun (2022). A Path Towards Autonomous Machine Intelligence.
5. Archimedes Project. github.com/aidless/agi-research.
6. Wang (2013). Non-Axiomatic Logic. World Scientific.
