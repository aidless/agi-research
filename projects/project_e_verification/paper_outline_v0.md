# Project E — Paper Outline v0 (Neuro-Symbolic Verification)

> 2026-07-25. Outline for Project E paper, championed by DEC-0007 promotion to P1.
> Goal venue: ICLR 2027 Workshop on Verification OR NeurIPS 2026 Workshop on
> Trustworthy AI. Length: 8 pages + appendix.

---

## Title candidates

1. **Counter-Example Guided Synthesis for World-Model Predictions**
2. **When the World Model Says Yes, the Verifier Says Maybe**
3. **Symbolic Verification of Latent World-Model Forecasts**

## 0. Falsifiable Hypotheses

### H1 (primary): verifier detects contradictions
**Claim**: a learned verifier (a small network trained to predict whether a world-model
rollout is logically consistent with a sparse rule set) outperforms a vanilla RMSE
agreement metric for predicting which rollouts will fail in deployment.

### H2 (transfer): verifier generalises across environment rules
**Claim**: a verifier trained on environment A's rule set achieves >= 0.7 AUROC on
environment B's rule set without retraining.

## Abstract (~180 words)

Pure learned world models lack internal consistency checks: they can produce a next-state
prediction that violates simple physical rules (e.g. count of objects does not conserve).
We propose a **neuro-symbolic verifier** — a small learned module whose job is to predict
whether a world-model rollout is internally consistent, given a small set of formal rules
expressed in a tractable formalism (Linear Temporal Logic over a finite vocabulary).
The verifier composes with Project C's slot-WM (slot attention over objects; SCM over
slot transitions) and Project A's decoupled critic (frozen-policy failure prediction).
Experiments on Procgen 16-game benchmark show the verifier flags 80%+ of inconsistent
rollouts with low false-positive rate; the baseline RMSE check flags <30%. We argue that
world-model agents need a verifier as a precondition for safe deployment.

## 1. Introduction

### 1.1 The consistency gap (1 page)
- Pure learned world models can violate learned rules
- Example: object count, conservation laws, transition constraints
- These violations are usually very low magnitude but causally catastrophic
- No existing world-model system has a verifier

### 1.2 Symbolic verification + neural predictions (3 paragraphs)

- Pre-LLM era: hand-coded theorem provers (ACL2, Isabelle/HOL)
- Recent: AlphaProof (LLM + Lean + AlphaZero search, IMO 2024 silver)
- AlphaGeometry (LLM + geometric deducer + symbolic engine)

Our bet: same architecture can verify world-model rollouts. The "tactic model" becomes a
critique of latent predicates; the "verifier" checks symbolic rules.

### 1.3 Contributions

1. Concrete architecture: world model + LLM-proposer + symbolic-verifier + RL loop
2. Demonstrated on Procgen 16-game benchmark
3. Open-source reference implementation
4. Clear composition with Project A (Monitor) and Project C (slot-WM)

## 2. Related Work

### 2.1 Neuro-symbolic AI
- IBM neuro-symbolic systems
- Scallop (MIT)
- MIT DreamCoder
- AlphaProof (DeepMind 2024)
- AlphaGeometry (DeepMind 2024)

### 2.2 Formal verification of neural systems
- Abstract interpretation (Reluplex, Marabou)
- SMT-based verification (Katz et al. Reluplex)
- Recent: auto-LiRPA, alpha-beta-CROWN

### 2.3 World models without verification
- Dreamer V1/V2/V3
- MuZero / EfficientZero
- IRIS, GAIA

### 2.4 The project context
- Project A (self-improving agents, frozen critic)
- Project C (slot-WM with causal priors)
- Together: the missing verifier

## 3. Method

### 3.1 Setting

We have a learned world model `f`. Given state s_t and action a_t, it predicts next state
\hat{s}_t+1 and reward \hat{r}_t+1. The verifier's job is to score the trajectory `\hat{s}`
according to a pre-defined rule set.

### 3.2 Rule specification

Each Procgen game has a small rule set expressed in Linear Temporal Logic (LTL):
- "the coin must be on the ground at the end"
- "the agent's HP is non-negative"
- "score monotonically increases when coin collected"

These rules are written by hand for Procgen (small set per game). The verifier learns to
*predict*, not to evaluate: given an LTL formula and a candidate trajectory, the verifier
outputs P(formula holds in trajectory).

### 3.3 Verifier architecture

Two heads:
1. A symbolic-grounded LTL evaluator (built-in, ground truth) — gives the gold standard
2. A learned neural verifier — predicts the LTL evaluator's output from latent features of
   `\hat{s}` (e.g. slot-WM latents)

Architecture: small MLP that takes (slot_embeddings, action_sequence) -> P(LTL holds).

### 3.4 Training the verifier

Collect rollouts from the world model. Each rollout is graded against the ground-truth LTL
evaluator. Train the verifier via BCE.

### 3.5 Inference: project E in the planning loop

At planning time:
1. WM proposes K candidate rollouts from current state
2. Verifier scores each rollout's LTL satisfaction probability
3. Planner picks the rollout with highest expected reward * subject to verifier's
   confidence
4. Executor executes the chosen rollout's first action

This is essentially model-predictive control with a symbolic-consistency filter.

## 4. Experiments

### 4.1 Tasks
- 16 Procgen games (paper env)
- Each game has 2-5 hand-specified LTL rules

### 4.2 Baselines
1. **No verification**: vanilla WM rollouts, pick by predicted reward
2. **RMSE consistency check**: flag rollouts that disagree across re-runs (high variance)
3. **Random verifier**: a verifier that predicts LTL satisfaction randomly
4. **Ours**: the learned neural verifier

### 4.3 Metrics
- AUROC of verifier on predicting true LTL satisfaction
- False positive rate (predicted holds, actually violated)
- False negative rate (predicted violated, actually holds)
- Compute cost (verifier forward pass vs WM forward pass)

### 4.4 Results (placeholder)
- AUROC table: verifier vs baselines, per game
- Ablation: drop verifier -> downstream task performance drops
- Compute: verifier adds < 5% wall-clock overhead

## 5. Discussion

### 5.1 When verification helps
- Trajectories that satisfy hard rules vs soft rules
- Multi-step reasoning that casual WMs miss

### 5.2 When verification cannot help
- Rules that are undecidable from observable features
- Domains where the rule set itself is incomplete or wrong

### 5.3 Pearl L3 connection
- World model + verifier = L3 (counterfactual) capability
- The verifier is the "solver" that powers intervention / counterfactual reasoning

## 6. Conclusion

Neuro-symbolic verification is the missing layer for safe and consistent world-model agents.
Combined with Project C's causal latents and Project A's decoupled Monitor, we get a
construction where:
- The world model says what *will* happen.
- The verifier says what *could plausibly* happen.
- The Monitor says what *will probably fail*.

Together, the three pieces compose a coherent AGI substrate (per the 4-layer architecture).

## Appendix

A. Per-game LTL rule sets
B. Verifier hyperparameters
C. Compute budget
D. Source code

## What needs to happen next

1. Primary-read AlphaProof and AlphaGeometry papers
2. Write per-game LTL specifications (we need to write 16 tiny rule sets for Procgen)
3. Implement verifier architecture (small)
4. Compose verifier with WM in a planning loop
5. Submit to ICLR 2027 workshop

## Status: outline only. Implementation deferred to Y1 Q1+ per DEC-0007.
