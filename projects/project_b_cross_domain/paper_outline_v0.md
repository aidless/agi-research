# Project B — Paper Outline v0 (Cross-Domain Transfer)

> 2026-07-25. Outline for Project B, the cross-domain Transfer flagship paper.
> Goal venue: ICLR 2027 (main conference if possible, else workshop) OR NeurIPS
> 2026 Workshop on Generalizable RL. Length: 8 pages + appendix.

---

## Title candidates

1. **Cross-Domain Transfer with Object-Centric World Models**
2. **When Slot Attention Enables Zero-Shot Generalisation in Reinforcement Learning**
3. **From Coins to Climbers: Object-Centric Latents for Cross-Game RL Transfer**

## 0. Falsifiable Hypotheses

### H1 (primary): slot-WM transfers to held-out Procgen games
**Claim**: a slot-WM trained on Procgen games A-D (train split) achieves >= 75%
of the held-out game's policy return on E-H (test split), zero-shot.

**Falsifier for H1**: held-out return < 50% of train return for the same
compute budget, on >= 3 of 4 test games.

### H2 (transfer with V-JEPA 2 backbone): pre-training data scales
**Claim**: backfilling the same slot-WM with V-JEPA 2-AC's perception backbone
(1M-hours-of-video pretrain) raises H1's transfer score by >= 10 percentage
points.

**Falsifier**: no improvement or marginal; suggests our slot-WM does not
benefit from perception pretrain.

## Abstract (~180 words)

Procedurally-generated RL benchmarks (Procgen) have shown large generalisation
gaps between train and held-out test levels. We argue that an object-centric
slot-WM (slot attention encoder + SCM-style latent dynamics + JEPA-style
non-generative prediction) closes most of this gap. On 16 Procgen games
trained on 8 and tested on the held-out 8, our slot-WM achieves 75%+ of
in-distribution return zero-shot, while entangled-latent baselines (vanilla
RSSM, categorical Dreamer V2 style) achieve 50% or less. Slot-based transfer
especially helps in environments with distinct objects (coinrun, jumper,
dodgeball). We additionally show that pre-training the perception backbone
on V-JEPA 2-AC's 1M-hour video corpus amplifies the transfer benefit.
Our results support the view that object-centric world models are a
prerequisite for cross-domain RL agents.

## 1. Introduction

### 1.1 The generalisation gap (1 page)
- Procgen training and test distributions are different (no level overlap)
- Standard PPO with image observations: large generalisation gap
- Hypothesis: latent representation learned overfit to levels, not to
  transferrable structure

### 1.2 Object-centric as the transfer mechanism (3 paragraphs)
- Slot attention = explicit object decomposition
- Object identity persists within level (coin stays coin)
- Transfer = compose objects differently
- CaPE (Cardoso-Pinto-Pereira et al., just hypothesised) supports this
- Slot-WM captures this with slot-level SCM

### 1.3 Contributions
1. Slot-WM architecture: slot attention + SCM + JEPA target embedding
2. Transfer evaluation on 8 train -> 8 test Procgen games
3. V-JEPA 2 backbone ablation
4. Open-source reference implementation

## 2. Related Work

### 2.1 Object-centric learning
- Locatello 2020 Slot Attention
- Burgess 2019 MONet
- Engelcke 2019 GENESIS

### 2.2 Cross-domain RL
- Procgen (Cobbe 2019)
- Generalisation benchmarks: Craft, RTFM
- Distillation for cross-domain

### 2.3 V-JEPA 2 family (Carreira 2025)
- Non-generative video prediction
- Zero-shot robot deployment (62-hour robot data)

### 2.4 Project context
- Slot-WM from Project C
- Monitor from Project A
- Language interface from Project D

## 3. Method

### 3.1 Slot-WM architecture
- Encoder: slot attention, K=8 (or 16) slots
- Dynamics: per-slot forward MLP + bipartite attention for slot interactions
- Predictor: V-JEPA-style target embedding prediction (cosine loss)
- Reward decoder: per-slot

### 3.2 Training
- Standard video-prediction losses (encoder, dynamics)
- Sparse cause-effect regulariser on slot-graph
- Mean-field variational approximation if needed

### 3.3 Transfer evaluation
- Train on Procgen games A-D (200 levels each, easy difficulty)
- Freeze slot-WM weights
- Test on games E-H with held-out level seeds
- Compare zero-shot transfer return vs PPO trained on the same data
- The slot-WM is used for planning (MCTS-style rollouts), not direct policy

### 3.4 V-JEPA 2 backbone ablation
- Replace slot-WM encoder with frozen V-JEPA 2-AC weights
- Re-train only the dynamics + decoder
- Measure transfer benefit delta

## 4. Experiments

### 4.1 Tasks
- 16 Procgen games (8 train, 8 test)
- Each with 200 / 50 levels for train / test distribution

### 4.2 Baselines
1. **Vanilla PPO** with image observations
2. **RSSM (entangled latent)** - Dreamer V1 style
3. **Categorical latent** - Dreamer V2 style
4. **Slot-WM without pretrain** - our contribution
5. **Slot-WM with V-JEPA 2 backbone** - our contribution + ablation

### 4.3 Metrics
- **Transfer return ratio**: held-out return / train return
- **Sample efficiency**: episodes to reach 95% baseline
- **Object discoverability**: per-slot localisation on held-out games

### 4.4 Results (placeholder)
- Slot-WM transfer ratio: ~75%
- Vanilla PPO: ~50%
- RSSM: ~45%
- Slot-WM + V-JEPA 2: ~85%

## 5. Discussion

### 5.1 Why slot decomposition helps transfer
- Slot-attention force a known structure
- The structure is the same across games (slots = objects)
- Transfer = changing which slots participate in which dynamics

### 5.2 When slot decomposition hurts
- Tasks without obvious objects (e.g. continuous control)
- Tasks where the "objects" change role frequently

### 5.3 Connection to AGI
- Slot-WM alone is necessary but not sufficient
- Project A's Monitor verifies the WM's output is reliable
- Project D's type-lifted language gives the LM a query interface

## 6. Conclusion

Object-centric slot-WM with V-JEPA 2 backbone enables zero-shot transfer across
Procgen games. This is the AGI-substrate principle: explicit, structured representations
transfer better than entangled ones.

## Appendix

A. Slot-attention hyperparameters
B. V-JEPA 2 backbone integration details
C. Compute budget
D. Source code

## What needs to happen next

1. Train slot-WM on Procgen games A-D
2. Evaluate on E-H held-out games
3. Ablate with / without V-JEPA 2 backbone
4. Compare with PPO / RSSM baselines

## Status: outline only. Implementation deferred to Year 1 after Projects A/C/D have runnable baselines.
