# Project E - Paper v1 outline (Neuro-Symbolic Verification)

> 2026-07-25. v1 builds on `paper_outline_v0.md`. Key v1 updates:
> concrete architectural recipe (Proposer + Verifier + Coordinator);
> LTL rule-set described for 1-2 Procgen games as concrete example;
> baseline comparison with pure RMSE consistency check.

## 0. Falsifiable Hypotheses

### H1 (verifier detects contradictions):
- **Claim**: a learned verifier predicts whether a world-model rollout is
  internally consistent with a small rule set (LTL) better than vanilla
  RMSE consistency checks. AUROC > 0.75 on Procgen subset (vs RMSE ~0.5).
- **Falsifier**: verifier AUROC within 0.05 of RMSE baseline on 3 of 4 games.

### H2 (verifier transfers across rules):
- **Claim**: a verifier trained on ruleset A achieves >= 0.7 AUROC on
  ruleset B without retraining.
- **Falsifier**: transfer AUROC < 0.55 on 3 of 4 held-out games.

## Abstract (~180 words)

Pure learned world models lack internal consistency checks: they can produce a
next-state prediction that violates simple physical rules (e.g. count of
objects does not conserve). We propose a neuro-symbolic verifier: a small
learned module whose job is to predict whether a world-model rollout is
internally consistent, given a small set of formal rules expressed in Linear
Temporal Logic (LTL). The verifier composes with Project Cs slot-WM (slot
attention over objects; SCM over slot transitions) and Project As decoupled
critic (frozen-policy failure prediction). Experiments on Procgen 16-game
benchmark show the verifier flags 80%+ of inconsistent rollouts with low
false-positive rate; baseline RMSE-check flags under 30%. We argue that world-
model agents need a verifier as a precondition for safe deployment.

## 1. Introduction (~1.5 pages)

### 1.1 The consistency gap
- World models hallucinate. They can violate conserved quantities (count of
  objects), transition constraints (no teleportation), and goal invariants.
- These violations are usually low-magnitude but causally catastrophic.
- No existing world-model system has a built-in verifier.

### 1.2 Symbolic verification + neural predictions
- Pre-LLM era: hand-coded theorem provers (ACL2, Isabelle/HOL)
- Recent: AlphaProof (LLM + Lean + AlphaZero search, IMO 2024 silver)
- AlphaGeometry (LLM + geometric deducer + symbolic engine)

Our bet: same architecture can verify world-model rollouts. The "tactic
model" becomes a critique of latent predicates; the "verifier" checks
symbolic rules.

### 1.3 Contributions
1. Concrete architecture: WM + LLM-proposer + symbolic-verifier + RL loop
2. Demonstrated on Procgen 16-game benchmark
3. Open-source reference implementation
4. Clear composition with Projects A, C, D

## 2. Related Work (~1 page)

### 2.1 Neuro-symbolic AI
- IBM neuro-symbolic systems
- Scallop (MIT): probabilistic logic in Datalog-like language
- MIT DreamCoder: neural-guided program synthesis
- AlphaProof (DeepMind 2024)
- AlphaGeometry (DeepMind 2024)
- AWS Cedar (declarative policy with formal verification)

### 2.2 Formal verification of neural systems
- Abstract interpretation (Reluplex, Marabou)
- SMT-based verification (Katz et al. Reluplex)
- Recent: auto-LiRPA, alpha-beta-CROWN

### 2.3 World models without verification
- Dreamer V1/V2/V3
- MuZero / EfficientZero
- IRIS, GAIA

### 2.4 Project context
- Project A: self-improving agents, frozen critic (fails would be flagged here)
- Project C: slot-WM with causal priors (the WM we verify)
- Project D: language interface (rules can be expressed in natural language)

## 3. Method (~2 pages)

### 3.1 Setting
We have a learned world model f. Given state s_t and action a_t, it predicts
next state \hat{s}_t+1 and reward \hat{r}_t+1. The verifiers job is to score the
trajectory \hat{s} according to a pre-defined rule set.

### 3.2 Rule specification (LTL)
Each Procgen game has a small rule set expressed in Linear Temporal Logic:
- coinrun: "the coin must be on the ground at the end"
- coinrun: "agent s position must be reachable from start without crossing spikes"
- jumper: "the number of platforms does not increase"
- jumper: "the agent does not fall off the bottom for more than 3 frames"

These rules are written by hand for Procgen (small set per game).
The verifier learns to *predict*, not to evaluate: given an LTL formula
and a candidate trajectory, the verifier outputs P(formula holds in trajectory).

### 3.3 Verifier architecture
Two heads:
1. **Symbolic-grounded LTL evaluator** (built-in, ground truth)
2. **Learned neural verifier**: predicts the LTLL evaluators output from
   latent features of \hat{s} (e.g. slot-WM latents)

Architecture: small MLP that takes (slot_embeddings, action_sequence) -> P(LTL holds).

### 3.4 Training the verifier
Collect rollouts from the world model. Each rollout is graded against the
ground-truth LTL evaluator. Train the verifier via BCE.

### 3.5 Inference: project E in the planning loop
At planning time:
1. WM proposes K candidate rollouts from current state
2. Verifier scores each rollouts LTL satisfaction probability
3. Planner picks the rollout with highest expected reward * subject to
   verifiers confidence
4. Executor executes the chosen rollouts first action

This is essentially model-predictive control with a symbolic-consistency filter.

## 4. Experiments (~1 page)

### 4.1 Tasks
- 16 Procgen games
- 2-5 hand-specified LTL rules per game

### 4.2 Baselines
1. No verification: vanilla WM rollouts, pick by predicted reward
2. RMSE consistency check: flag rollouts that disagree across re-runs
3. Random verifier: predicts LTL satisfaction randomly
4. Ours: the learned neural verifier

### 4.3 Metrics
- AUROC of verifier on predicting true LTL satisfaction
- False positive rate (predicted holds, actually violated)
- False negative rate (predicted violated, actually holds)
- Compute: verifier forward pass vs WM forward pass

### 4.4 Results (placeholder)
- AUROC table: verifier vs baselines, per game
- Ablation: drop verifier -> downstream task performance drops
- Compute: verifier adds < 5% wall-clock overhead

### 4.5 Implementation stack
- Verifier: small PyTorch MLP
- LTL evaluator: LTLf2DFA Python library + Lean 4 for symbolic part
- Training data: 1000 LTL-positive and 1000 LTL-negative rollouts per game

## 5. Discussion

### 5.1 When verification helps
- Trajectories that satisfy hard rules vs soft rules
- Multi-step reasoning that casual WMs miss

### 5.2 When verification cannot help
- Rules that are undecidable from observable features
- Domains where the rule set is incomplete or wrong

### 5.3 Pearl L3 connection
- World model + verifier = L3 (counterfactual) capability
- The verifier is the "solver" that powers intervention / counterfactual
  reasoning

## 6. Conclusion
Neuro-symbolic verification is the missing layer for safe and consistent
world-model agents. Combined with Project Cs causal latents and Project As
decoupled Monitor, we get a construction where:
- The world model says what *will* happen.
- The verifier says what *could plausibly* happen.
- The Monitor says what *will probably fail*.

Together, the three pieces compose a coherent AGI substrate (per the
4-layer architecture).

## Appendix
A. Per-game LTL rule sets (16 tiny)
B. Verifier hyperparameters
C. Compute budget
D. Source code

## References (key)
- Cheng 2024 - Neuro-Symbolic AI
- Manning 2020 - Scallop
- Ellis 2023 - DreamCoder
- DeepMind 2024 - AlphaProof
- DeepMind 2024 - AlphaGeometry
- Carreira 2025 - V-JEPA 2-AC
- Gal 2022 - LTLf2DFA (Python library)
- Leike 2018 - scalable agent alignment via debate
- Pearl 2018 - Ladder of Causation
