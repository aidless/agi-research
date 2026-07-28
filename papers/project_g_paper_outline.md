# Project G Paper Outline -- LLM Self-Monitoring via Decoupled Monitors

> Date: 2026-07-28
> Status: Outline only. Real paper depends on H10 + H11 results.
> Target venue: NeurIPS 2027 (May 2027 submission) or EMNLP 2027
> Length target: 12-14 pages (8 main + 4-6 appendix)

---

## Title (working)

"Decoupled Monitors for Self-Rewarding LLM Agents: A Pre-Registered
Study of the Decoupling Principle Beyond Classical RL"

or, shorter:

"Does the Decoupled Monitor Principle Transfer from PPO to LLM
Self-Rewarding?"

## Abstract (~250 words)

We test whether the decoupled-Monitor principle -- validated on
classical RL with the H1 ablation (frozen Monitor > joint Monitor
on LunarLander-v3, 5/5 seeds, AUROC delta 0.724) -- transfers to the
LLM self-rewarding domain. We pre-register H10 (frozen vs joint
Monitor on frozen-LLM rollouts) and run it on a small LM (Qwen-1.5B
or Phi-3-mini) with GSM8K-style math reasoning traces.

The H10 experiment compares three arms -- frozen Monitor, joint
Monitor, and random Monitor (negative control) -- on n=5 seeds,
200 rollouts per seed. Failure labels are derived from final-answer
correctness on GSM8K ground truth. The Monitor architecture reuses
the Archimedes Slot-Monitor (Section 3.5) adapted to LLM trace input.

We complement this with H11, which tests cross-environment transfer
of the H10 finding to a different dataset and a different frozen LM.
H11 is contingent on H10: if H10 is REFUTED, H11 is moot.

We report all three arms'' results honestly. If H10 holds, the
decoupling principle is more general than classical RL. If H10 is
refuted, the principle is classical-RL-specific and the Y1.x + H2.0
closure''s null synthesis generalizes to LLMs.

## 1. Introduction (1 page)

### 1.1 Motivation

Self-improving LLM agents are an active research area (RLHF, DPO,
constitutional AI, self-rewarding loops). A common pattern is to
use a learned critic or reward model to provide auxiliary signal
during training. The standard approach -- jointly training the
critic with the policy -- has a known weakness: the critic''s
gradients get dragged by the policy update.

The Archimedes Project has documented this weakness in classical
RL: the H1 ablation shows that a frozen Monitor (trained on
rollouts from a frozen policy) achieves AUROC 0.796 while a joint
Monitor (trained jointly with PPO) achieves AUROC 0.072 -- worse
than random. The decoupling principle is real on classical RL.

### 1.2 The open question

Does the decoupling principle transfer to LLMs? If yes, the
principle is more general. If no, the principle is classical-RL-
specific.

### 1.3 Contributions

1. **H10 pre-registered study** of frozen vs joint Monitor on LLM
   traces (Section 4).
2. **H11 follow-up** testing cross-environment transfer of H10
   (Section 5).
3. **LLMSlotMonitor architecture**: the Slot-Monitor adapted to
   LLM traces (Section 3).
4. **Honest framing**: every positive result paired with stated
   limitations per NO_SELF_DECEPTION.md.

## 2. Background (1.5 pages)

### 2.1 Self-rewarding LLM agents

- RLHF (Christiano 2017, Ouyang 2022)
- DPO (Rafailov 2023)
- Constitutional AI (Bai 2022)
- Self-rewarding (Yuan 2024)
- Reflexion / Self-Refine / CRITIC (Shinn 2023, Madaan 2023, Gou 2024)
- PRM (Lightman 2023)

### 2.2 Decoupled critics in RL

- H1 ablation (Archimedes Project A, 5/5 seeds)
- CQL (Kumar 2020) -- frozen Q for evaluation, but training updates critic
- Frozen-critic variants

### 2.3 Slot attention for sequences

- Locatello 2020 (object-centric scene decomposition)
- Adaptations to 1-D trajectory sequences (Archimedes Project A,
  Y1.3 Slot-Monitor AUROC 0.989)

### 2.4 The Archimedes Project

5-year research program; relevant components: Slot-Monitor, DLR,
GovBench governance primitives. Project G extends Project A to
the LLM domain.

## 3. Method (3 pages)

### 3.1 Setup

- Frozen LM (Qwen-1.5B or Phi-3-mini; user choice)
- Reasoning dataset (GSM8K or MATH; user choice)
- Trace representation: slot attention on last 20 (token, logit) pairs
- Failure label: GSM8K final-answer correctness

### 3.2 LLMSlotMonitor architecture

Reuses Project A Slot-Monitor (4 slots, 32 dim, 3 attention
iterations), adapted to (token, logit) input pairs.

### 3.3 Three arms

1. **Frozen Monitor**: trained on rollouts from frozen LM. Test arm.
2. **Joint Monitor**: trained jointly with simulated LLM updates.
   Control arm.
3. **Random Monitor**: untrained random signal. Negative control.

### 3.4 Training procedure

- Frozen: 50 epochs, Adam lr=1e-3, BCE loss.
- Joint: 10 LLM steps x 5 Monitor epochs, perturbation scale 0.05.
- Random: no training.

### 3.5 Evaluation

- 50 held-out traces per seed
- AUROC on (failure_prob, is_failure) pairs
- Reported per-seed, then aggregate (mean, std, Welch t)

### 3.6 Pre-registered decision rule

See `experiments_log/2026-07-28-PRE-REGISTERED-H10.md`.

## 4. H10 results (3 pages)

### 4.1 Per-seed results

Table: 5 seeds x 3 arms (frozen, joint, random). AUROC per cell.

### 4.2 Aggregate

Mean, std, Welch t for each arm. Pre-registered decision rule
applied.

### 4.3 Verdict

H10 VALIDATED / REFUTED / INCONCLUSIVE.

### 4.4 Mechanism discussion

If H10 is VALIDATED: what mechanism explains the decoupling
advantage in LLMs? Is it the same as in classical RL (joint
Monitor learns a *policy-coupled* failure concept)?

If H10 is REFUTED: what explains the negative result? Is the
failure concept different in LLMs (more reasoning-based, less
trajectory-based)?

## 5. H11 cross-environment results (2 pages, contingent on H10)

### 5.1 Setup A: New dataset

If H10 is VALIDATED, repeat on MATH (or other formal-reasoning
dataset).

### 5.2 Setup B: New LM

If H10 is VALIDATED, repeat on Phi-3-mini (or other small LM).

### 5.3 Verdict

H11 VALIDATED / REFUTED / INCONCLUSIVE. Both setups must pass for
VALIDATED.

## 6. Discussion (1.5 pages)

### 6.1 If H10 is VALIDATED

- The decoupling principle is more general than classical RL.
- Implications for self-rewarding LLM agents: training-time use of
  frozen Monitor signal may be a useful auxiliary.
- Caveats: single LM, single dataset, n=5.

### 6.2 If H10 is REFUTED

- The decoupling principle is classical-RL-specific.
- The Y1.x + H2.0 closure''s null synthesis generalizes: Monitor
  signal is useful for offline analysis (verifier) but not for
  online training intervention.
- Implications: LLM self-rewarding should use joint Monitor
  (no advantage to decoupling) or different primitives entirely.

### 6.3 Cross-cutting discussion

- The Monitor as verifier vs the Monitor as reward signal (the
  Archimedes "verifier, not reward" theme from Y1 paper v3.7).
- Slot attention''s role: did it help in the LLM domain?
- Failure-concept stability: how does the joint Monitor''s failure
  concept compare to the frozen Monitor''s?

## 7. Limitations and future work (1 page)

### 7.1 Single LM / single dataset

The H10 result is on one LM and one dataset. H11 tests transfer
but with a small number of setups.

### 7.2 No GPU

Small LMs may be too slow on CPU for a thorough study. The user
needs to decide whether to use GPU or accept slower iteration.

### 7.3 No real LLM in smoke test

The smoke test (h10_smoke.py + joint_monitor.py) uses synthetic
data, not real LLM rollouts. The real H10 experiment requires
a frozen LM.

### 7.4 Future work

- **H12**: Multi-agent LLM rollouts (extending H5 to LLM domain).
- **H13**: Self-improvement loop on LLM traces (extending H9).
- **H14**: DLR predicates on LLM reasoning traces (Project E
  meets Project G).

## References (1 page)

~30 references, including all LLM self-rewarding work, classical
RL frozen-critic work, slot attention work, and Archimedes
Project references.

## Appendices (online-only)

- A: Per-seed detailed metrics for all 3 arms
- B: Hyperparameter reference
- C: Code index
- D: Reproducibility instructions
- E: Synthetic-data smoke-test results (for transparency)

---

*Outline prepared 2026-07-28. Real paper draft depends on H10 +
H11 results. Outline will be revised once H10 verdict is known.*