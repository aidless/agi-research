# Writing Sample Outline — Archimedes H1 Ablation

> PhD Application Materials, Version 1.0
> Date: 2026-07-27
> Source: Thesis v1.0 Part II Project A (Chapter 6)

---

## Paper Title

**Decoupled Monitors: A Mechanism for Stable Self-Monitoring in
Reinforcement-Learning Agents**

## Abstract (200 words)

A self-improving agent must be able to predict its own failures. We argue
that jointly training the failure-prediction Monitor with the policy destroys
its discrimination power: the Monitor''s gradients get pulled by the policy
update, reducing its ability to discriminate failure vs non-failure
trajectories. We propose **decoupling**: train the Monitor on rollouts from
a *frozen* policy, never updating it during policy training.

We validate this hypothesis on LunarLander-v3, training PPO for 100K
steps and comparing a frozen-policy Monitor to a jointly-trained Monitor
across 5 random seeds. The frozen Monitor achieves mean AUROC 0.796
(range 0.21-0.98); the joint Monitor achieves mean AUROC 0.072 (range
0.041-0.103). The mean delta is 0.724, with 5/5 seeds supporting
decoupling (Wilcoxon signed-rank p = 0.0625 one-sided).

We extend this to **Slot-Monitor**: replacing the raw-history Monitor
input with slot-attention features improves AUROC from 0.796 to 0.989
(+0.193, 24% relative), demonstrating that structural decomposition
amplifies the decoupling advantage.

We then test the policy-action implications: 6 follow-up experiments
attempting to use the Monitor signal for inference-time intervention
all failed. However, a recent **training-time regularization** approach
(Y1.3, by concurrent session) produced the first positive result:
shaping PPO rewards with `Monitor_prob(window)` improved mean return
by +50 over baseline (3/5 seeds). This suggests decoupled Monitors are
valuable as **constraints during learning**, not as direct interventions.

## 1. Introduction (3 pages)

- Motivation: self-monitoring is critical for AGI safety.
- Problem: joint training destroys discrimination.
- Contribution: H1 ablation validates decoupling (5/5 seeds).
- Bonus: Slot-Monitor structural improvement (+0.193 AUROC).
- Discussion: inference-time vs training-time use of Monitor.

## 2. Background and Related Work (2 pages)

- Self-critics and STaR-family methods.
- Frozen-critic baselines (CQL).
- Slot attention (Locatello 2020).
- ENWI framework theoretical foundation.

## 3. Method (3 pages)

- MDP formulation for LunarLander.
- Monitor architecture (3-layer MLP).
- Frozen Monitor training procedure.
- Joint Monitor training procedure.
- Slot-Monitor adaptation.

## 4. Results (4 pages)

- H1 5-seed ablation table.
- Statistical analysis (Wilcoxon, bootstrap CI).
- Slot-Monitor +0.193 improvement.
- Per-seed analysis (seed 2 anomaly discussion).
- Inference-time intervention attempts (6 failures, DEC-0011 HALT).
- Training-time regularization (Y1.3, +50 result).

## 5. Discussion (2 pages)

- Why decoupling helps (stationary data distribution).
- Why joint training fails (covariate shift).
- Inference-time vs training-time: when each works.
- Limitations (LunarLander only, single env).
- Future work (cross-env, multi-seed verification).

## 6. Conclusion (1 page)

## Appendices

- A: Per-seed detailed metrics
- B: Hyperparameter reference
- C: Code index
- D: Reproducibility instructions

## References (1 page)

~30 references

---

*Total target: ~15-18 pages. This is a typical NeurIPS / ICML paper length.*

## Notes for Application

- The writing sample should be the **abstract + §1 + §3 + §4** of this
  outline, ~8-10 pages.
- The most novel contribution is §4.1 (H1 5/5 seeds).
- The bonus contribution is §4.2 (Slot-Monitor +0.193).
- The honest discussion is §4.4 + §5 (inference-time intervention fails).

## Adapt for Application

1. Match page limit (most programs require 5-25 pages).
2. Trim §4 to focus on H1 ablation; cut §4.4-§4.5 to discussion.
3. Include the abstract as a 1-page summary if needed.
4. Cite the Archimedes repo (github.com/aidless/agi-research) for reproducibility.
