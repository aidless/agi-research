# GAE - Generalised Advantage Estimation (Schulman et al. 2016)

> Date read: 2026-07-25 (from Codex training-data memory)
> Confidence: **HIGH** -- classic, well-documented
> One-line takeaway: GAE is an exponentially-weighted average of n-step
> returns that smoothly interpolates between TD(0) bias and MC variance.

---

## Problem

Policy-gradient methods need an advantage estimator `A(s,a)`. Two
extremes exist:
- TD(0): low variance, high bias
- Monte Carlo return: zero bias, high variance

Practitioners want a knob.

## Method

```
A_t^GAE(gamma, lambda)
   = sum_{l=0}^{infinity} (gamma * lambda)^l * delta_{t+l}

where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

`lambda = 0` recovers TD(0); `lambda = 1` recovers MC return.

Implementation: back-up through the rollout computing `delta_t`,
accumulate into `A_t^GAE`. Standard PPO uses `gae_lambda = 0.95` and
`gamma = 0.99`.

## Empirical result

The original GAE paper (ICML 2016) shows:
- 5x faster wall-clock convergence on MuJoCo with same hyperparameters
  vs TD(0)
- Enables training RNN-policies (asynchronous advantage actor-critic,
  the A2C at the time) on continuous control

Compute cost: negligible on top of any on-policy algorithm.

## Criticisms

1. GAE assumes a good value function `V`. If `V` is wrong (especially
   under-trained), GAE compounds the error.
2. `gae_lambda = 0.95` is reasonable for most tasks but is a real
   hyperparameter for tasks with very long horizons -- review our
   Project A monitor design for this.
3. There are alternatives (e.g. V-trace from IMPALA) that handle
   off-policy correction; GAE is purely on-policy.

## Connection to our program

Our `code/ppo.py` uses GAE. We should:
- Verify `gae_lambda = 0.95` is reasonable for Procgen 16-game benchmark
- Document the choice in Project A paper appendix
- Potentially ablate `gae_lambda in {0.0, 0.5, 0.95, 1.0}` and report
  decoupling effect at each

## Confidence

HIGH. Standard, well-documented. Re-verify on:
- exact mu / lambda defaults in modern implementations
- the connection to A3C where GAE was popularised

## Related papers

- A3C / A2C (Mnih 2016)
- IMPALA / V-trace (Espeholt 2018)
- Retrace / Tree-backup (various)

## Status

- [x] code reproduces
- [ ] ablating gae_lambda added to Project A ablations
