# AlphaProof (DeepMind 2024 IMO Silver)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (post-training-cutoff claim; based on official
> DeepMind announcements)
> One-line: Combined LLM-based formal-language reasoning (Lean) with
> RL-trained AlphaZero-style search. Solves IMO-level problems; won
> silver medal at IMO 2024 (combined with AlphaGeometry for geometry).

---

## Problem

Formal mathematical proof is hard for ML systems. Lean (proof assistant)
requires precise formal manipulation; LLMs hallucinate; pure search is
too slow.

AlphaProof bet: combine LLM intuition (Lean tactics) + AlphaZero-style
RL on proving policy.

## Method

Two-stage:

1. **Pretrain a "tactic model"** on large corpus of human-written Lean
   proofs. This LLM suggests the next tactic given the proof state.

2. **RL in the proving environment**:
   - State = current proof state (sequence of Lean tactics + goal)
   - Action = next tactic
   - Reward = 1 if proof complete, 0 otherwise
   - Use a variant of AlphaZero's MCTS to explore proof trees in
     the tactic space

The trick: **formal-language RL with linguistic semantic priors**.
The LLM provides informed proposals; the search verifies and
optimises.

## Empirical result

IMO 2024 (held July 2024):
- AlphaProof solved 4 of 6 problems (P1, P2, P4, P5? I think)
- Combined with AlphaGeometry for geometry: 4/6
- Score: 28/42 = P6 was missed; scored partial on others
- Silver medal threshold met; not gold (gold requires 5+ correct or
  harder thresholds)

This is the **first time an AI system scored near the IMO medal
threshold**.

## Criticisms

1. **Specialised to mathematics**. The same architecture doesn't
   directly transfer to general reasoning.

2. **Compute cost is substantial**: thousands of TPU days reportedly
   per IMO problem variant.

3. **The problem distribution is curated**. AlphaProof was fine-tuned
   on (human-written + self-generated) Lean proofs of similar style.

4. **It does not solve open problems**. It solves competition problems
   with known short proofs.

5. **Reproducibility is limited**. DeepMind hasn't released full code;
   results are not independently verified at scale.

## Connection to our program

**AlphaProof is the strongest existing proof-of-concept for our
"neuro-symbolic" verification layer (Project E)**:

- AlphaProof = LLM (semantic priors) + formal verifier (Lean)
  + AlphaZero-style search
- This is the **concrete recipe** for Project E: a learned component
  generates hypotheses, a symbolic verifier checks them, an RL loop
  improves the generator over time.

We can position Project E in our program as:
"Generalise AlphaProof from mathematics to general reasoning tasks.
Use a world model as the lean equivalent (the formal language), and
LLM-as-tactic-model for hypothesis generation."

## Related papers

- Lean theorem prover (de Moura 2015)
- AlphaZero (Silver 2018) - the RL framework
- AlphaGeometry (DeepMind 2024 - geometry-specific)
- Llemma (2023): Lean-pretrained LLM

## Status

- [x] cite in TASKBOOK Project E description
- [x] cite as Project E's motivating example
- [ ] mark Project E as P1 (your DEC-007 priority decision pending)

## Confidence

MEDIUM. Announced publicly by DeepMind at IMO 2024. Full paper may
have come out around or after IMO (I recall a Nature paper but
uncertain). User should primary-read the actual paper before citing.
