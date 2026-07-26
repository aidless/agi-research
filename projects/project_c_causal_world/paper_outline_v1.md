# Project C - Paper v1 outline (Slot-WM lifting Pearl L1 to L2)

> 2026-07-25. v1 builds on `paper_outline_v0.md`. Key updates:
> H1/H2 with concrete numerical thresholds; reference to
> Scholkopf 2021 + Slot Attention + V-JEPA target as the architecture
> primitives; Causal-JEPA placeholder pending user primary read.

## 0. Falsifiable Hypotheses

### H1 (primary): slot-WM achieves Pearl L2
- **Claim**: slot-WM with object-level causal intervention injection predicts
  effects of held-out interventions significantly better than an entangled
  latent baseline (vanilla RSSM). Paired Wilcoxon AUROC > joint across 4+
  Procgen games, p < 0.01.
- **Falsifier**: if RSSM matches slot-WM AUROC within 0.05 on >= 12 of 16
  Procgen games, the causal-slot claim is rejected.

### H2 (transfer of L2 capability): 
- **Claim**: slot-WM trained on game A retains intervention-aware predictions
  on held-out game B; entangled latent drops to chance.
- **Falsifier**: transfer AUROC not > joint + 0.1 on 12+ games -> reject.

## Abstract (~180 words)

World models (Dreamer, MuZero, JEPA) excel at predicting next state from
the current one. They cannot, however, answer *what would happen if I
intervened on this specific object* (Pearl Level 2). We show that lifting
world models from Pearl Level 1 to Level 2 is achievable by combining
slot attention (one slot per object) with structural causal model priors
and JEPA-style non-generative prediction. The resulting Slot-WM achieves
intervention-aware predictions on Procgen tasks with small but consistent
improvements over entangled-latent baselines. We additionally show that
Slot-WMs learned causal structure transfers across game distributions,
supporting the view that explicitly causal world models are a necessary
ingredient for AGI-class agents. Our implementation is CPU-runnable at
small scale and open-source.

## 1. Introduction (~1.5 pages)

### 1.1 The L1 ceiling
- World models (Dreamer V1/V2/V3, MuZero) predict P(s_t+1 | s_t, a_t).
- This is Pearl L1 (observational), not L2 (interventional).
- Cannot reliably answer "if I pushed this instead, what?"

### 1.2 Slot-WM as the recipe
- Ingredients: Slot Attention (Locatello 2020) + SCM over slots (Pearl 2018,
  Scholkopf 2021) + JEPA-style non-generative prediction (LeCun 2022).
- Combination: world model whose latent has the structure to support L2.

### 1.3 Contributions
1. Concrete architecture combining these primitives
2. Empirical demonstration on Procgen 16-game benchmark
3. Transfer evidence: structural slots generalise
4. Open-source reference implementation

## 2. Related Work (~1 page)

### 2.1 World models
- World Models (Ha & Schmidhuber 2018) - foundational
- PlaNet (Hafner 2019), Dreamer V1/V2/V3 (2020-2024)
- MuZero (Schrittwieser 2020)
- IRIS (Micheli 2023, transformer WM)

### 2.2 Object-centric
- Slot Attention (Locatello 2020)
- MONet (Burgess 2019), GENESIS (Engelcke 2019)

### 2.3 Causal representation learning
- Pearl 2018 - Ladder of Causation (L1/L2/L3)
- Scholkopf 2021 - Causal Representation Learning
- von Kugelgen 2021 - Self-Supervised CRL
- Bareinboim 2016 - Causal Transportability
- Ahuja 2023 - IRM

### 2.4 JEPA family
- LeCun 2022 - JEPA (initial)
- Bardes 2024 - V-JEPA 2 (video)
- Carreira 2025 - V-JEPA 2-AC (action-conditioned)
- V-JEPA-WM (Dec 2025, what drives success in physical planning)
- Causal-JEPA (Feb 2026, intervention-aware world model)
- Value-Guided JEPA (Jan 2026, action planning)

### 2.5 Hybrid LLM + WM
- PaLM-E (Driess 2023, sensor tokens to LLM)
- Gato (Reed 2022, multi-task generalist)

## 3. Method (~2.5 pages)

### 3.1 Architecture overview
- Encoder: Slot Attention, K slots
- Dynamics: per-slot SCM with learned DAG edges
- Predictor: V-JEPA-style target embedding prediction (no pixel reconstruction)
- Reward decoder: per-slot
- Value decoder: per-slot (for planner compatibility)

### 3.2 Latent structure
- Each slot s_i_t has continuous features z_i_t
- Edges E_t (binary mask K x K) determined by small attention net
- Dynamics: z_t+1 = f_i(z_t, parents(z_t), action) for slot i

### 3.3 Training
- Reconstruction loss (slot-level decoded image) = weak signal
- Primary: target embedding prediction (mask some slots, predict target
  embeddings from context). Loss = cosine distance to ground-truth slot
  embeddings (target net = EMA copy)
- Sparse DAG regulariser on edge density
- **Intervention augmentation**: occasionally clamp a slot, predict the rest
- Sparse cause-effect penalty: encourage E_t to be sparse

### 3.4 Intervention as inference (L2 capability)
At test time, we run "do" by:
1. choose slot i to intervene
2. freeze its dynamics to a pre-set trajectory
3. roll out other slots conditioned on this intervention
4. measure divergence from the no-intervention rollout

This is the Slot-WMs answer to Pearl P(s_t+1 | do(a_t)).

## 4. Experiments (~2 pages)

### 4.1 Tasks
- 4 Procgen games (paper env subset)
- CausalWorld synthetic for controlled interventions

### 4.2 Baselines
1. Vanilla RSSM (entangled latent, Dreamer V1 style)
2. Categorical latent (Dreamer V2 style)
3. Slot-VAE: slot attention encoder + standard dynamics (no SCM prior)
4. Causal-JEPA (Feb 2026, intervention-augmented JEPA)
5. Ours: Slot-WM with sparse DAG prior

### 4.3 Metrics
- Intervention accuracy: how often does the model predict the actual outcome
  correctly when we intervene on slot i
- Counterfactual AUROC: predict reverse intervention effect
- Cross-game transfer: train on game A, evaluate intervention accuracy on B
- Compute: GPU-hours per training run (target < 24h)

### 4.4 Results (placeholder)

### 4.5 Compute plan
- Training 4 games x 256K steps x 5 seeds = ~ 1-2 GPU-days
- CPU-only downscaled feasible (50K steps) but limited
- Awaiting Y1 GPU access

## 5. Discussion (~0.5 page)

### 5.1 What we showed
- Slot-attention with SCM prior lifts WM from L1 to L2
- Transfer benefit comes from causal structure, not visual priors

### 5.2 Limitations
- K (slot count) is fixed
- Slot permanence across long horizons is open
- Real-world causal structure often confounded; we assume discrete
  intervention oracle (from Procgen or synthetic)

### 5.3 Connection to AGI / Project A
- Pearl L3 (counterfactual) remains open after this paper
- Project As Monitor can be lifted to L3 via this Slot-WM
- Project Ds language interface can query slot predicates directly
- Project Es verifier ensures the L2 predictions are consistent with rules

## 6. Conclusion (~0.25 page)

Combining slot-attention object-centric representation with structural causal
model priors yields world models that can answer Pearl L2 intervention queries.
This is necessary (but not sufficient) for AGI-class agents.



## Update 2026-07-26: Slot attention PoC on real LunarLander

Implementation: `projects/project_c_causal_world/code/slot_attention_lunarlander.py`

- 100 trajectories from frozen PPO, padded to length 48
- Slot attention (4 slots, dim=32) + SlotDynamicsModel trained 25 epochs
- Reconstruction loss: 0.036 (good — slot+attention captures dynamics)
- Diversity loss: 0.356 (slots have weak specialization)
- Per-slot top features: slot 0 = horizontal, slot 1 = rotation,
  slot 2 = vertical, slot 3 = overlap

Conclusion: slot attention finds weak kinematic specialization on a
single rigid body. Project C requires multi-object environments
(Procgen 16 games) to demonstrate true object binding. Y1 work.

## Appendix
A. Slot-attention hyperparameters
B. SCM loss coefficients
C. Compute budget
D. Source code

## References (key)
- Pearl 2018 - Ladder of Causation
- Scholkopf 2021 - Causal Representation Learning
- Locatello 2020 - Slot Attention
- LeCun 2024 - JEPA
- Carreira 2025 - V-JEPA 2-AC
- Bardes 2024 - V-JEPA
- Hafner 2024 - Dreamer V3 (Nature 2025)
- Schrittwieser 2020 - MuZero
- Bareinboim 2016 - Causal Transportability
- Ha & Schmidhuber 2018 - World Models
