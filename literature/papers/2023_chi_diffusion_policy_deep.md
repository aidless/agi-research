# Diffusion Policy (Chi et al. 2023, RSS 2024)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- recent, well-cited, code released
> One-line: A conditional denoising diffusion model that, conditioned on the
> recent observation history, denoises a chunk of action trajectories; flagship
> embodied-AI approach for visuomotor control.

## Problem this solves
Visuomotor control (predicting robot actions from camera + proprioception)
has many failed approaches:
- Behaviour cloning is data-hungry.
- RL is sample-inefficient.
- Sequence Transformers (ACT, RT-1) are good but slow at inference.
- Diffusion models have shown strong generative capabilities elsewhere (image, video).
Why not actions?

## Method
Action chunking + diffusion. The model:
- **Input**: observation history (RGB + proprio, last few frames)
- **Output**: a chunk of H future actions (e.g., H=8 steps)
- **Inference**: denoise Gaussian noise for K steps (e.g., K=10) to get a clean action sequence.
- **Loss**: standard DDPM loss on the action sequence.

Crucially: action SEQUENCES are multi-modal. For "pick the apple" there are
many valid grasping trajectories; diffusion models natively model multi-modal
distributions better than Gaussian.

## Empirical result
- 5 simulated tasks (Push-T, Lift, etc): SOTA on most.
- 6 real-world tasks: 5/6 SOTA, average success rate +46.9% over previous SOTA.
- Inference ~3 Hz on RTX 4090; real-time control.
- Multi-task: single model trained on 50+ tasks works.

## Criticisms (specific)
1. **No explicit planning**: the diffusion model samples action chunks, not plans.
2. **No uncertainty representation in policy** (just samples).
3. **Action horizon is fixed**: unable to handle long-horizon tasks directly.
4. **Compute cost**: tens of denoise steps per inference.

## Connection to our program
Diffusion Policy is a competing architecture for Project B (cross-domain):
- We propose slot-WM + V-JEPA 2 backbone. They propose diffusion action chunking.
- Both deal with action prediction from observation.
- Their diffusion gives them a multi-modal action distribution;
  our slot-WM gives us counterfactual reasoning (L2 capability).

For Project A:
- Diffusion Policy has no failure predictor. But its action chunks could
  be evaluated by a Monitor.
- Integration: Diffusion Policy for action proposal, Monitor (Project A)
  for failure assessment, retry on Monitor alert.

## Confidence
HIGH.

## Related
- DDPM (Ho 2020), DDIM (Song 2021)
- Decision Transformer (Chen 2021)
- ACT (Zhao 2023) - action chunking transformer
- Stable Diffusion (Rombach 2022) - diffusion image foundation
- Open X-Embodiment (Collaboration 2023) - multi-robot data
- RT-1 / RT-2 (Google)
- V-JEPA 2-AC (Carreira 2025) - alternative for cross-embodiment

## Status
- cited in Project B Related Work (alternative cross-domain approach)
- cited in Project A future (Monitor over Diffusion Policy chunks)
- future: hybrid diffusion + slot-WM + Monitor as Project B
