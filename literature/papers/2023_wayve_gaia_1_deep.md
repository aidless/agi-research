# GAIA-1 (Hu et al. Wayve 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: A generative world model for autonomous driving. Given video
> history, action sequence, and text prompt, GAIA-1 generates plausible
> future video frames conditioned on those inputs.

## Problem
Autonomous driving needs a model of how the world behaves. Pure RL from
pixels is too sample-inefficient; hard-coded rules are brittle; pure video
prediction ignores action. GAIA-1 unifies them.

## Method
A video diffusion model conditioned on:
- Past video frames (video history)
- Action sequence (planned or hypothetical)
- Text prompt (style / scene description)

Generates future frames. The driving policy is then a separate module on
top of GAIA-1s outputs (look-ahead planning + control).

Training: large-scale driving video, paired with CAN-bus action logs.
Architecture: transformer-based diffusion.

## Empirical result
- Drives on standard benchmarks (Wayve internal + nuScenes-related).
- Predicts multi-second rollouts that look realistic to human drivers.
- Policy trained atop GAIA-1 rollouts improves over video-only baseline.
- Closest comparable: NVIDIA DriveDreamer (2023), Tesla FSD planner (closed).

## Criticisms (specific)
1. **No action conditioning in the loop during deployment** - policy is
   extrinsic.
2. **Closed-world assumption**: extreme conditions (e.g. flooded road)
   may be out of distribution.
3. **Compute cost**: high. Not edge-deployable.
4. **Goal-conditioned, not goal-aware**: no Pearl-style intervention
   capability.

## Connection to our program
For Project B (cross-domain transfer):
- GAIA-1 is the analog of our slot-WM but for driving video, not games.
- Comparison: does slot-attention help in driving video like it does in
  Procgen?

For Project C (causal WM):
- Driving is messy; causal structure is partially observable (other cars
  are hidden causes of behaviour).
- GAIA-1 surface-level video prediction; Project C wants cause-aware
  prediction.

## Confidence
HIGH.

## Related
- V-JEPA 2 (Bardes 2024) - non-generative video prediction
- V-JEPA 2-AC (Carreira 2025) - action-conditioned
- NVIDIA DriveDreamer (2023)
- Genie 2 (DeepMind 2024)
- Cosmos (NVIDIA 2025)
- Wayve Lingo (driving commentator)

## Status
- cited in Project B Related Work (driving world model comparator)
- cited in Project C Related Work (cause vs appearance)
- future: compare to current SOTA in autonomous driving benchmarks
