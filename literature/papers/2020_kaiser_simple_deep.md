# SimPLe (Kaiser et al. 2020, DeepMind)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- explicit DeepMind paper that became the basis for Procgen
> One-line: A first-of-its-kind world-model RL agent that, with 100K environment
> frames, matches DQN that uses 100x more. This same paper's "hard distribution" insight
> directly motivated the Procgen benchmark.

---

## Problem

Atari games had been studied with model-free RL for years, but world-model agents
like World Models 2018 were either domain-specific or used too many env steps.
A clean benchmark: how sample-efficient can a world-model RL be on Atari?

## Method

SimPLe = Simple Player Learning. Specifically:
- Autoencoder for image representation
- Linear dynamics model in latent space (yes, LINEAR)
- PPO on rollouts in latent space
- Conservative loss on dynamics to avoid model collapse

The paper's contribution is NOT the algorithm itself but the empirical
demonstration: with this simple stack, 100K Atari frames match 100x that for DQN.

## Empirical result

- 100K frames of environment interaction
- vs DQN baseline using 4-40M frames
- Median human-normalised score: similar to DQN at far less data
- Some games (Asteroids, SeaQuest) worse than DQN; others (Pong) much better

## Criticisms (specific)

1. **Linear dynamics is restrictive**. Games with non-linear dynamics (e.g., swinging pendulums,
   skill jumps) underperform.

2. **The VAE reconstruction can be confused with "predicting" success**. SimPLe's policy
   might be learning to recognise game-state in the VAE latent, not learning good dynamics.

3. **Hard-mode vs easy-mode distractor**. The paper's distribution-shift observation (easy
   mode games don't transfer to hard mode) inspired Procgen directly. This is the paper's
   most lasting contribution: identifying that the *generalisation* question is the real
   challenge in model-based RL.

4. **Compute on world model is non-trivial**. Each iteration requires training the VAE,
   the dynamics, and the policy. Wall-clock can exceed model-free methods even if env
   interactions are fewer.

## Connection to our program

SimPLe is **the most direct intellectual antecedent of Procgen**.

Cobbe et al. 2019 designed Procgen after observing SimPLe's hard-mode distribution shift.
Our Project A's paper env (DEC-0008) is Procgen precisely because:
- SimPLe showed generalisation matters more than sample efficiency
- Procgen was designed to measure generalisation across distribution-shift

So this paper is **the chain link** from sample efficiency to cross-game transfer.

## Related

- World Models (Ha 2018)
- Dreamer V1 (Hafner 2020)
- IRIS (Micheli 2023)
- Procgen (Cobbe 2019, our paper env)
- Dreamer V3 (Hafner 2025)

## Status

- [x] cite in Project A paper Section 2 / Project B Section 2
- [x] include the "Hard mode generalisation is the real challenge" point in paper intro
- [x] pre-cursor citation for our Procgen paper-env choice
