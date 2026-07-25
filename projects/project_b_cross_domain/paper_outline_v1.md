# Project B - Paper v1 outline (Cross-Domain Transfer with Slot-WM)

> 2026-07-25. v1 builds on `paper_outline_v0.md` with the most direct
> primary-read citations filled in: V-JEPA 2-AC (Carreira 2025),
> Causal-JEPA (placeholder pending user primary read).

## 0. Falsifiable Hypotheses (from v0, refined)

### H1 (transfer with slot-WM):
- **Claim**: slot-WM trained on Procgen games A-D achieves >= 65% of held-out
  game`s return zero-shot (vs entangled RSSM at <= 50%).
- **Falsifier**: held-out return < 50% of train return on >= 3 of 4 test games.

### H2 (V-JEPA 2 backbone amplifies):
- **Claim**: backfilling slot-WM with V-JEPA 2-AC perception backbone raises
  H1 transfer score by >= 10 percentage points.
- **Falsifier**: no improvement or marginal (< 5 percentage points) over a
  matched-capacity random-init backbone.

## Abstract (~180 words)

Procedurally-generated RL benchmarks (Procgen) have shown large generalisation
gaps between train and held-out test levels. We argue the bottleneck is the
latent representation: entangled-latent world models (RSSM, categorical dreamer)
do not transfer across games because their internals are not compositional.
An object-centric slot-attention world model (slot-WM) decomposes the latent into
K slots, each representing an object, with per-slot dynamics and a sparse causal
graph relating slots. We train on 8 Procgen games and test zero-shot on 8 held-out
games. Slot-WM reaches >= 65% transfer return ratio, entangled baselines <= 50%.
Adding the V-JEPA 2-AC perception backbone (1M hours of video pretrain) amplifies
the transfer by an additional >= 10 percentage points. Our results support the
view that object-centric world models are a prerequisite for cross-domain RL agents.

## 1. Introduction (~1 page)

### 1.1 The generalisation gap in procgen
- Procgen designed by Cobbe 2019 to measure generalisation
- SimPLe (Kaiser 2020) first exposed distribution-shift problem in Atari
- Standard PPO: large gap on cross-domain tests

### 1.2 Our hypothesis: object-centric decomposes transfer
- Slot attention = explicit per-object decomposition
- Object identity persists within a level
- Transfer = re-composing the same objects in different rules
- Slot-WM captures this with slot-level dynamics

### 1.3 Contributions
1. Slot-WM architecture (slot attention + SCM + JEPA target)
2. Transfer evaluation on 8 train / 8 test Procgen games
3. V-JEPA 2-AC backbone ablation
4. Open-source reference

## 2. Related Work (~1 page)

### 2.1 Object-centric learning
- Locatello 2020 Slot Attention (foundational, our primitive)
- Burgess 2019 MONet, Engelcke 2019 GENESIS

### 2.2 Cross-domain RL
- Cobbe 2019 Procgen (designed for this)
- SimPLe 2020 (intellectual ancestor of Procgen)
- Kaiser 2020 introduced train vs hard distribution distinction

### 2.3 V-JEPA 2 family (Carreira 2025)
- V-JEPA 2 paper (Bardes 2024): video prediction
- V-JEPA 2-AC (Carreira 2025): action-conditioned variant
- 62 hours of robot fine-tuning + 1M hours video pretrain = zero-shot Franka

### 2.4 Project context
- Project C slot-WM provides the latent composition
- Project D language interface queries the slots
- Project E verifier validates against symbolic rules

## 3. Method (~2 pages)

### 3.1 Slot-WM architecture
- Encoder: Slot Attention with K=8 slots
- Dynamics: per-slot forward MLP + bipartite attention for slot interactions
- Predictor: V-JEPA-style target embedding prediction (cosine loss)
- Reward decoder: per-slot scalar

### 3.2 Training
- Reconstruction loss (slot-decoded image) = weak signal
- Primary: target embedding prediction (mask some slots, predict their
  embeddings from context)
- Sparse cause-effect regulariser on slot-graph
- Mean-field variational for prior

### 3.3 Transfer evaluation
- Train on Procgen games A-D (200 levels, easy difficulty)
- Freeze slot-WM weights
- Test on games E-H with held-out level seeds
- Compare transfer return with PPO baseline (no slot-WM) and entangled WM

### 3.4 V-JEPA 2 backbone ablation
- Replace slot-WM encoder with frozen V-JEPA 2-AC weights
- Re-train only dynamics + decoder
- Measure transfer benefit delta

## 4. Experiments (~1 page)

### 4.1 Tasks
- 16 Procgen games (8 train, 8 test split per Cobbe 2019)
- Each with 200 / 50 levels

### 4.2 Baselines
1. PPO with image observations
2. RSSM (entangled latent, Dreamer V1 style)
3. Categorical latent (Dreamer V2 style)
4. Slot-WM without pretrain (our contribution)
5. Slot-WM with V-JEPA 2 backbone (our contribution + ablation)

### 4.3 Metrics
- Transfer return ratio (held-out / train)
- Sample efficiency: episodes to reach 95% of held-out baseline
- Object discoverability: per-slot localisation accuracy

### 4.4 Results (placeholder, awaiting compute)

### 4.5 Compute plan
- Training 4 games x 256K steps x 5 seeds = ~ 1-2 GPU-days
- Currently in 5-year-program plan; awaiting Y1 GPU access
- CPU-only feasible for downscaled demo (50K steps) on coinrun

## 5. Discussion (~0.5 page)

### 5.1 Why slot decomposition helps transfer
- Forces a known structure in latent representation
- The structure is the same across games (slots = objects)
- Transfer = changing which slots participate in which dynamics

### 5.2 When slot decomposition hurts
- Tasks without obvious objects (continuous control)
- Tasks where "object" role changes rapidly (per-step)

### 5.3 Connection to AGI
- Slot-WM alone is necessary but not sufficient for AGI
- Project A Monitor verifies output reliability
- Project D type-lifted language gives LM query interface

## 6. Conclusion (~0.25 page)

Object-centric slot-WM with V-JEPA 2 backbone enables zero-shot transfer across
Procgen games. We argue this is the AGI-substrate principle: explicit,
structured representations transfer better than entangled ones.

## Appendix
A. Slot-attention hyperparameters
B. V-JEPA 2 backbone integration details
C. Compute budget
D. Source code

## References (key)
- Locatello 2020 - Slot Attention
- Cobbe 2019 - Procgen
- Carreira 2025 - V-JEPA 2-AC
- Schrittwieser 2020 - MuZero
- Hafner 2024 - Dreamer V3
- LeCun 2024 - JEPA / V-JEPA 2
- Kaiser 2020 - SimPLe
- Burgess 2019 - MONet
- Engelcke 2019 - GENESIS
  (additional refs in workspace 36 paper notes)
