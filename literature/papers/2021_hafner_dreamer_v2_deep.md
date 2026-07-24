# Dreamer V2 (Hafner et al. 2021, ICLR 2021)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: First world-model-based RL agent to reach human-level on
> Atari by replacing Gaussian stochastic latents with discrete
> categorical latents.

---

## Problem

Dreamer V1 used Gaussian stochastic latents. Empirically this caused
"posterior collapse" on high-variance image tasks like Atari: the
policy gradient signal was wasted; latents didn't carry useful info.

## Method

Change to the latent:
- replace Gaussian stochastic h_t with categorical h_t (one-hot per slot)
- the categorical prior + posterior are learned as above
- the **rewards at every step** are explicitly predicted by the model

Architectural diff vs V1:
- "Recurrent State-Space Model with discrete latents"
- K x 32 categorical latents per step (so 32 categorical variables each
  with 32 categories = roughly equivalent to 32*5 bit latent)

This discrete representation empirically gives better signal for
decision-relevant features because it forces the latent to commit.

## Empirical result

- Atari: superhuman median; comparable to Rainbow DQN on 55 of 55 games
- DMControl: comparable to or better than V1

A single Dreamer V2 model trained with no per-game tuning reached
superhuman level on 53 of 55 Atari games.

## Criticisms

1. **Discrete latents lose smoothness**. Continuous interpolation in
   latent space becomes problematic. Downstream tasks that need smooth
   control may suffer.

2. **Categories are unbounded in count**. Choosing 32x32 was empirically
   tuned; there's no principled derivation.

3. **Atari-specific quirks**. Some Atari games have very sparse rewards
   (Montezuma's Revenge). Dreamer V2 didn't help much there -- still
   an exploration problem.

4. **Compute scaling still non-trivial**. Several GPU-days per run.

## Connection to our program

V2 informs our Project C choice:
- Categorical latents are a step toward identifiable, object-like
  latents (combined with slot attention)
- The V1 -> V2 progression is the pattern: simple -> structured
- Our eventual target is structurally-identified object latents with
  causal relations; V2's categorical is a primitive along the path.

## Related

- Dreamer V1 (2020)
- Dreamer V3 (2024/2025)
- RSSM (Hafner 2019)
- PonderV2 (2023): Transformer world model on similar ideas

## Status

- [x] cite in Project C Section 3 (architectural progression)
