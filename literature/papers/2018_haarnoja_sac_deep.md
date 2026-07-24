# SAC - Soft Actor-Critic (Haarnoja et al. 2018 / 2019)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Maximum-entropy off-policy RL. Reformulates RL objective to
> jointly maximise reward and policy entropy, yielding a stable, sample-
> efficient algorithm that has become the de-facto off-policy baseline.

---

## Problem

Prior off-policy methods (DDPG, TD3) had stability issues on continuous-
control tasks with high-DOF action spaces (humanoid, dexterous hand).

## Method

Objective: maximise expected discounted reward PLUS a Shannon-entropy
term weighted by temperature alpha:

```
J(pi) = sum_t E[ r(s_t, a_t) + alpha * H(pi(.|s_t)) ]
```

Implementation:
- two Q networks (clipped double Q, like TD3)
- stochastic policy updated by minimising KL between policy and exponential
  of Q
- automatic temperature tuning: alpha adjusted so entropy stays near a
  target (default = -dim(action))

Algorithm is off-policy, replay buffer, batched.

## Empirical result

- 21 of 22 MuJoCo continuous-control tasks solved from scratch
- Sample-efficient: 5x faster than PPO on humanoid
- Stable across hyperparameters (compared to TD3)
- Compatibility with HER (Hindsight Experience Replay) - widely used for
  manipulation tasks

## Criticisms

1. **Maximum-entropy on sparse-reward envs can be slow to converge**. The
   policy explores well but doesn't exploit good policies enough; can
   take many gradient updates.

2. **Temperature alpha is a crucial hyperparameter**. Auto-tuning helps
   but can be unstable.

3. **Off-policy buffer size is large**. Millions of transitions typical.

4. **Policy randomness (entropy) can hurt deployment**. After training,
   you typically want a deterministic policy. SAC gives you a stochastic
   policy that averages out to deterministic only at the limit.

5. **Doesn't compose with priority sampling**. Some extensions attempted
   but SAC + priority replay has subtle instability.

## Connection to our program

SAC is **background knowledge** for Project A but not primary:
- on-policy PPO is cleaner for the decoupling claim
- SAC would require a separate "off-policy monitor" experiment
- but we should cite SAC in Project A background to acknowledge we
  considered it

## Related

- TD3 (Fujimoto 2018) - the immediate predecessor
- PPO (Schulman 2017) - on-policy alternative
- Soft Q-Learning (Haarnoja 2017) - SAC conceptual ancestor
- RLPD (2022) - SAC + demonstrations for offline+online

## Status

- [x] cite in Project A Related Work (background)
- [ ] not in primary method
