# ACT - Action Chunking Transformer (Zhao et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: A transformer that, conditioned on current+past visual observations,
> autoregressively predicts a chunk of H future actions. Introduces temporal
> ensembling at inference; precursor to Diffusion Policy.

## Problem
Visuomotor control: predict robot actions from images. Standard BC:
- single-step prediction is myopic (mistakes compound)
- closed-loop action regression fails on long-horizon tasks

## Method
ACT predicts action chunks (H=8 steps, e.g.) and only executes the first k,
then re-predicts. The chunking amortises prediction cost; re-prediction
catches mistakes early.

Inference variant: temporal ensembling with overlapping chunks, weighted by
how recently predicted.

## Empirical result
- Real-robot pick + place: ~75-85% success on 6 tasks, beating single-step
  BC by 30+ points.
- Inference ~3 Hz on A100 - real-time control.
- Method is simple enough to implement in 200 lines (some implementations).

## Criticisms (specific)
1. **Action chunk is independent of environment**: predicting next chunk
   does not condition on intermediate feedback.
2. **Failure recovery is brittle**: once the chunk goes wrong, recovery is
   only by re-prediction.
3. **No semantic understanding**: actions are values, not types.

## Connection to our program
For Project B (cross-domain):
- ACT is a competing architecture with Diffusion Policy.
- Diffusion Policy improves on ACT by replacing Gaussian action regression
  with diffusion sampling.

For Project A (Monitor + Project B):
- ACT produces action chunks. A Monitor (Project A) could evaluate each
  chunk for safety before execution.
- Combined: Monitor + ACT = "generate action chunk then ask Monitor for
  approval".

## Confidence
HIGH.

## Related
- Behaviour cloning (BC, Pomerleau 1991)
- Diffusion Policy (Chi 2023) - downstream improvement
- RT-1 (Brohan 2022) - Google robot transformer
- Decision Transformer (Chen 2021)
- RoboNet (Dasari 2019)

## Status
- cited in Project B Related Work (action chunking baseline)
- cited in Project A future Monitor+chunks integration
