# Genie / Genie 2 (DeepMind 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **MEDIUM** (cutting edge, near my training cutoff)
> One-line: Interactive environment generators. Given a text prompt OR an
> image prompt, generate interactive 2D/3D environments controllable via
> learned latent actions. A foundation model for env generation.

## Problem
RL agents need many environments. Building Procgen itself required manual
work. Genie asks: can we LEARN an environment generator from video?

## Method
Two-stage:
- **Stage 1 (video model)**: spatio-temporal transformer over Internet 2D
  video games. Learns to predict next frame given past frames + latent action.
- **Stage 2 (latent action model)**: tokenise actions as discrete tokens in a
  VQ-VAE-style codebook; learned from video of action-following frames.

Inference:
- Given a text prompt (Genie 2 has text encoder), generate initial frame.
- Apply latent action tokens to evolve the scene.
- Use as a synthesised environment for downstream RL.

## Empirical result
- Genie (2024 first version): 2D platformer-style scenes; ~10B parameters.
- Genie 2 (2024 second): supports text prompting, longer-horizon interaction.
- Works as world model for downstream RL agents.

## Criticisms (specific)
1. **Generation quality varies**: many generated levels are nonsensical.
2. **No semantic understanding**: latent actions are statistical.
3. **Compute**: huge parameter count.
4. **Hard to validate** - how do we test if a generated environment is
   useful?

## Connection to our program
Genie maps to multiple projects:
- **Project B (cross-domain)**: Genie could generate Procgen-style envs as
  data augmentation.
- **Project C (causal WM)**: Genie is an alternative to MuZero / Dreamer
  for learned world models.
- **Project D (language types)**: Genie 2 supports text-to-env generation,
  an alternative to LLM-as-type-system.

## Confidence
MEDIUM.

## Related
- Genie (DeepMind 2024)
- Genie 2 (DeepMind 2024 second)
- Sora (OpenAI 2024) - similar video generation, no env control
- Cosmos (NVIDIA 2025) - alternative foundation world model
- Genie 3 (DeepMind 2025) - reported real-time interactive 3D generation
- MineWorld (OpenAI 2025) - similar project

## Status
- cited in Project B Related Work (env generation alternative)
- cited in Project C Related Work (alternative world model approach)
- future: compare Cosmos vs Genie 3 vs Dreamer V3 for our use case
