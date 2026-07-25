# Dreamer V3 (Hafner et al. 2024 / Nature 2025)

> Date: 2026-07-25 deep-read from training-data memory, near edge of cutoff
> Confidence: **MEDIUM**
> One-line: First world-model-based RL agent to reach SOTA on 150+ diverse tasks
> with a *single set of hyperparameters*, including Minecraft diamond zero-shot.

## Problem
Prior world-model agents needed per-task tuning of entropy, GAE lambda,
replay ratio, network size. For 150+ tasks this is infeasible.

## Method
Four bets: (1) Symlog/symexp encoding for all signals; (2) Free-bits KL
regulariser stops posterior collapse; (3) Critic ensemble K=2 with quantile
regression; (4) Continuous-action normalising flows instead of Gaussian.

## Empirical result
- 150+ tasks at SOTA with one config
- Minecraft: first WM agent to find a diamond zero-shot
- DMControl / Atari: comparable or better than V2

## Criticisms (specific)
1. The no-tuning claim is overstated -- task-specific discrete-vs-continuous
   choices are made.
2. Symlog is leaky for small reward ranges (essentially linear there).
3. Normalising flows hurt reproducibility.
4. Humanoid MuJoCo: some early reports suggest Dreamer V3 did not fully solve.
5. Minecraft diamond is partly a pipeline feat.

## Connection to our program
Engineering reference for Project C, future benchmark for Project A.
- Symlog for reward is a generic trick; use it.
- Free-bits KL on per-slot latents is a structural prior.
- Normalising flows matter for continuous control.
- Monitor ABOVE Dreamer V3 is a future Project A experiment.

## Confidence
MEDIUM. Headlines publicised; ablation numbers I am uncertain on. Need
primary read for paper-grade citation.

## Status
- [x] cite as Project C engineering reference
- [x] cite as Project A future experiment target
- [ ] primary read required for paper-grade citation
