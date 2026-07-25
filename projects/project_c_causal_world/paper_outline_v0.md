# Project C — Paper Outline (v0 draft)

> Goal: NeurIPS 2026 workshop OR ICLR 2027 workshop.
> Length: 8 pages main + unlimited refs + 4 page appendix.

---

## Title (top 3 candidates)

1. **Causal-JEPA on Object Slots: Lifting World Models from Prediction to Intervention**
2. **Lift RSSM to Pearl L2: Slot-Wise SCM with Procgen Empirical Evidence**
3. **Causally-Informed Latent Dynamics: A Hybrid Slot-WM Benchmark**

## 0. Falsifiable Hypotheses

### H1 (primary): causal latents improve monitor signal

**Claim**: a world model with structured slot-level SCM latents (our Slot-WM)
predicts the effect of interventions in held-out Procgen levels significantly
better than an entangled-latent baseline (vanilla RSSM); paired Wilcoxon
AUROC > joint across 4+ games, p < 0.01.

**Falsifier for H1**: if RSSM matches Slot-WM AUROC within 0.05 on >= 3
of 4 Procgen games, the causal-slot claim is rejected.

### H2 (transfer): causal latents generalise across environments

**Claim**: a Slot-WM trained on game A retains AUROC > 0.6 on held-out
game B; entangled-latent drops to ~0.5. The intervention structure
is what transfers.

**Falsifier for H2**: transfer AUROC not > joint latent + 0.1 on 3+ games
-> rejection.

## Abstract (~180 words)

> World models (Dreamer, MuZero, JEPA) excel at predicting next state from
> the current one. They cannot, however, answer *what would happen if I
> intervened on this specific object* (Pearl Level 2). We show that lifting
> world models from Pearl Level 1 to Level 2 is achievable by combining
> slot attention (one slot per object) with structural causal model priors.
> The resulting Slot-WM achieves intervention-aware predictions on Procgen
> tasks with small but consistent improvements over entangled-latent
> baselines. We additionally show that Slot-WM's learned causal structure
> transfers across game distributions --- supporting the view that
> explicitly causal world models are a necessary ingredient for AGI-class
> agents. Our implementation is CPU-runnable at small scale and open-source.

## 1. Introduction (~1.5 pages)

### 1.1 The L1 Ceiling (3 paragraphs)
- WM achieve L1 (predict next state)
- They cannot answer Level 2 (intervention) without structure
- Argued by Pearl, Bareinboim: anything that *only* learns P(s_t+1 |
  s_t, a_t) cannot answer P(s_t+1 | do(a_t))

### 1.2 Slot-WM as the Recipe (3 paragraphs)
- Slots = objects (Locatello 2020)
- SCM over slots (Pearl 2018, Scholkopf 2021)
- V-JEPA-style non-generative prediction (LeCun 2022)
- Combined = a world model whose latent has structure suitable for L2

### 1.3 Contributions
1. Concrete architecture combining these primitives
2. Empirical demonstration on Procgen 16-game benchmark
3. Transfer evidence: structural slots generalise
4. Open-source reference implementation

## 2. Related Work (~0.5 page)

- World models: World Models (2018), PlaNet, Dreamer V1/V2/V3
- Object-centric: Slot Attention (Locatello 2020), IODINE, MONet
- Causal: Pearl Ladder, Causal-JEPA (2026), Scholkopf CRL
- JEPA family: V-JEPA, V-JEPA 2-AC
- Hybrid LLM+WM: PaLM-E

## 3. Method (~2.5 pages)

### 3.1 Architecture overview
- Encoder: Slot Attention, K slots
- Dynamics: per-slot SCM, edges from learned DAG
- Predictor: V-JEPA-style embedding prediction (no pixel reconstruction)
- Reward decoder: per-slot
- Value decoder (if needed for planner): per-slot

### 3.2 Latent structure
- Each slot s_i_t has continuous features z_i_t
- Edges E_t (binary mask of K x K) determined by small attention net
- Dynamics: z_t+1 = f_i(z_t, parents(z_t), action) for slot i

### 3.3 Training
- Reconstruction loss (slot-level decoded image) = weak signal
- **Primary**: V-JEPA-style target embedding prediction
  - mask some slots, predict their target embeddings from context
  - loss = cosine distance to ground-truth slot embeddings (target net = EMA)
- Sparse DAG prior loss: regulariser on edge density
- Intervention augmentation: occasionally clamp a slot's update, ask the
  model to predict the rest

### 3.4 Intervention as inference
At test time, we run "do" by:
- choose slot i to intervene
- freeze its dynamics to a pre-set trajectory
- roll out other slots conditioned on this intervention
- measure divergence from the no-intervention rollout

This is the Slot-WM's answer to Pearl P(s_t+1 | do(a_t)).

## 4. Experiments (~2 pages)

### 4.1 Tasks

**Paper env**: 4 Procgen games (coinrun, bigfish, jumper, dodgeball)
**Dev env**: CartPole-v1, LunarLander-v2

### 4.2 Baselines
1. **Vanilla RSSM**: Dreamer V1-style latent dynamics (entangled)
2. **Categorical latent**: Dreamer V2-style (still entangled)
3. **Slot-VAE**: slot attention encoder + standard dynamics (no SCM prior)
4. **Ours**: Slot-WM (slot attention + SCM prior + V-JEPA target)

### 4.3 Metrics
- **Intervention accuracy**: P(actual intervention effect ~ predicted) per slot
- **Counterfactual AUROC**: predict reverse intervention effect
- **Cross-game transfer**: train on game A, evaluate intervention accuracy on B
- **Compute**: GPU-hours per training run

### 4.4 Results (placeholder - to be filled)
- Table: per-game intervention accuracy (4 games x 3 seeds)
- Figure: cross-game transfer heat map
- Ablation: drop SCM prior -> transfer drops; drop slot attention -> intervention disappears

## 5. Discussion (~0.5 pages)

### 5.1 What we showed
- Slot attention with SCM prior lifts WM from L1 to L2
- Transfer benefit comes from causal structure, not visual priors

### 5.2 Limitations
- K (slot count) is fixed
- Slot identity / permanence across long time horizons is open
- Real-world causal structure is often confounded; we assume discrete
  intervention oracle (from Procgen)

### 5.3 Connection to AGI
- Pearl L3 (counterfactual) remains open; we propose this as future work
- Hybridisation with language (Project D) for type system
- Project E (verification) is complementary

## 6. Conclusion (~0.25 page)

## Appendix

### A. Hyperparameters
### B. Compute budget
### C. Per-game detailed trajectories
### D. Slot count K sensitivity
### E. Source code link + reproduction commands

---

## Connection to existing program

This paper v0 outline exists alongside:
- Project A (Self-Improvement / decoupled Monitor) - lightweight / paper first
- Project D (Language as type system) - opens Y1
- Project E (Neuro-symbolic verification) - opens Y1

We commit to Project C paper as **primary** deliverable beyond Project A paper v1.

---

## What needs to happen next

1. Reach DEC-0011: scaling Step 2 of Phase 1 (256K * 3 seeds * 4 games)
2. Reach DEC-0012: Project A Monitor integration into Phase 2
3. Once we have GPU, this paper moves from outline to full draft
4. We need to verify all 4 cited 2025-2026 papers (Causal-JEPA, V-JEPA 2,
   JEPA-WM, Value-Guided JEPA) by reading them personally.
