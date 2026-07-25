# DreamerV3 Implementation Notes (Hafner et al. 2024 Nature 2025)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (high-level only; specific architecture details
  in the Nature paper need primary read)
> One-line: A single hyperparameter configuration works across 150+ tasks.

## What DreamerV3 does

A world-model-based RL agent that:
- learns a recurrent latent dynamics model from observations
- imagines trajectories in latent space
- learns the policy via backprop through the imagined rollouts
- achieves SOTA on 150+ tasks with no per-task tuning

## Key architectural elements
- **Discrete latent**: in DreamerV2 categorical latents worked better than
  Gaussian for high-dim observations.
- **Symlog / symexp encoding** of all signals. symlog compresses large
  values, leaves small ones linear.
- **Free bits KL**: minimum KL between prior and posterior, prevents
  collapse to deterministic.
- **Critic ensemble K=2** with quantile regression.
- **Normalising flows for the policy**. Inverted Gaussian is too restrictive;
  a flow can express multi-modal action distributions.

## Implementation (typical)
- 1-7 days training per task on single GPU.
- Inference: ~50 actions/sec.
- Reaches SOTA at ~50M env steps.

## What we want for our Program
- Implementation skeleton: encoder, RSSM, decoder, value head, policy head.
- Actor-critic on imagined trajectories.
- Symlog in reward prediction (cheap win).
- Free bits KL on our slot latents (when we extend to Project C).

## Connection to our program
Project C slot-WM inherits from DreamerV3 in spirit but adds:
- Causal structure (not just prediction)
- Object decomposition via slot attention
- Non-generative loss (target embedding not pixel)

For Project A (Monitor):
- DreamerV3-style free bits KL is a cheap architectural improvement to
  any future Monitor with latent state.

## Confidence
MEDIUM (high-level architecture public; specific numbers from Nature
paper needs primary read).

## Related
- Dreamer V1 (Hafner 2020)
- Dreamer V2 (Hafner 2021)
- IRIS (Micheli 2022)
- TD-MPC (Hansen 2022)
- DreamerV3-Nature-2025-claims

## Status
- cited in Project C Related Work (key prior)
- implementation skeleton: pending (Y2 work)
- Minecraft zero-shot diamond: key headline to verify in Nature paper
