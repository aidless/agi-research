# Dyna-Q (Sutton 1991)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **VERY HIGH** -- foundational paper, deep-learning-era Dyna builds on this
> One-line: Combine model-free Q-learning with planning on a learned model of
> the environment; the *Dyna* architecture imagines and learns simultaneously,
> making the loop closed.

## Problem this solves

Reinforcement learning needs many real env interactions. Learning is much
cheaper than interaction. Question: can we combine them so both improve
from the same data?

## Method

Dyna-Q loop runs two processes in parallel on the same experience:
1. **Direct RL**: tabular Q-learning update on real (s, a, r, s) tuples.
2. **Model learning**: update a learned model M(s, a) -> (r, s) for visited states.
3. **Planning**: run n-step Q-learning *updates against imagined rollouts*
   from M in the same Q-table.

Both updates feed the same Q-table. Same data, two learning roles.

## Empirical result

- Dyna-Q in maze and blocking-maze navigation: faster convergence than
  pure Q-learning.
- Particularly strong when the agent can plan the path through learned
  walls (which real env never showed).

## Criticisms (specific)

1. **Model bias is critical**. If the model is wrong, planning gives wrong
   answers. The world-models / Dreamer line of work tries to fix this with
   better neural models.
2. **Tabular**: scales poorly. Modern Dyna variants use function approximation
   (e.g. neural-network models).
3. **No latent structure**: model is a forward simulator, not an abstract
   representation. Limits transfer.

## Connection to our program

Dyna is **the conceptual grandparent of Dreamer / World Models / MuZero**.
Specifically:
- World Models paper (Ha 2018) cites Dyna explicitly as inspiration.
- Dreamer V1 (Hafner 2020) is essentially "Dyna with a deep neural model".
- MuZero is "Dyna where the model itself is a learned latent MCTS".
- I2A (Hamrick 2017) is "Dyna where rollouts go directly into the policy".

Project As decoupled Monitor can be interpreted as a Dyna-extension:
- Direct RL = PPO
- Model learning = slot-WM (Project C)
- Planning = the value-function-backed policy
- Monitor = an additional learned component over the model-driven plans.

## Confidence
VERY HIGH.

## Related
- Dyna-2 (Sutton 1995) - extending to multiple time scales
- Deep Dyna-Q (2018+) - function approximation
- World Models (Ha 2018) - Dyna with VAE
- Dreamer V1-V3 - Dyna with RNN
- MuZero - Dyna where model is itself searched by MCTS

## Status
- cited in Project C Related Work (the intellectual grandparent)
- cited in Project A (Dyna framework gives our 4-layer a stronger lineage)
