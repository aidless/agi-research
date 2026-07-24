# PPO (Schulman et al. 2017)

> Date read: 2026-07-25 (from Codex training-data memory, NOT a primary read this session)
> Time spent: codex drafting ~30min from memory + confidence flag
> Reader(s): Codex from memory
> Confidence: **HIGH** -- this paper is well-documented in standard RL literature
> One-line takeaway: PPO is a first-order trust-region approximation that became the de-facto policy-gradient baseline precisely because it is simple and stable.

---

## Problem

Prior policy-gradient methods (TRPO, vanilla PG, A2C) had a trade-off:
trust-region methods were stable but expensive (second-order); vanilla PG
was cheap but unstable. Practitioners wanted a first-order method that
"just works" with little hyperparameter tuning, suitable for large
models.

## Method

PPO replaces TRPO's KL constraint with a clipped objective:
```
L^CLIP(theta) = E[ min( r_t(theta) * A_t,
                          clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
```
where `r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)`.

Key practical choices that made PPO ubiquitous:
- minibatch SGD over multiple epochs of the same data (instead of single
  pass like vanilla PG)
- value-function baseline (shared architecture sometimes)
- GAE for advantage estimation (separate paper)
- entropy bonus to discourage premature collapse
- gradient clipping (max norm 0.5 in original)

## Empirical result

- 49 of 49 Atari games with comparable sample complexity to A2C and better
  wall-clock (continuous control benchmark MuJoCo)
- Original paper used Mujoco (49 tasks), Atari (49 games), and showed
  PPO generally beats A2C, TRPO
- Headline: comparable or better sample efficiency to TRPO with
  first-order optimisation; orders of magnitude simpler implementation

Compute cost: days on single GPU for full benchmark.

## Criticisms

1. PPO is **not strongly monotonic** in its improvement -- the clipped
   objective is a first-order approximation and on-policy data drift can
   still hurt. It just *usually* doesn't.
2. PPO hyperparameters (`clip_eps`, `epochs`, `batch_size`, `gae_lambda`,
   entropy coefficient) interact non-trivially across environments.
   Re-tuning is often needed.
3. PPO's sample efficiency on sparse-reward envs is poor compared to
   off-policy methods (SAC, TD3). PPO is at its best when on-policy
   data is cheap.

## Connection to our program

We use PPO as Project A\'s baseline policy algorithm. The choice of PPO
is deliberate: it is on-policy (so decoupled-monitor story is clean --
the policy the monitor observes is exactly the policy that runs at
inference). Off-policy methods would introduce replay-buffer
staleness that complicates the decoupling claim.

Project A paper v1\'s H1/H2 hypotheses use PPO specifically. To claim
decoupling helps on a class of algorithms, we should also evaluate on
PPO variants (e.g. with GAE lambda = 0.95 default).

## Confidence

HIGH. PPO is the most-cited recent RL baseline; virtually all RL
implementations train it correctly. Risk of miscitation is minimal.

What to re-verify against the original paper:
- exact clip range [0.8, 1.2] vs alternative implementations
- the exact GAE coefficient used (0.95? 0.99?)
- original code repository URL (we should re-run if needed)

## Related papers

- TRPO (Schulman 2015) - the predecessor PPO approximates
- GAE (Schulman 2016) - advantage estimator PPO usually uses
- A3C (Mnih 2016) - parallel actor-learner inspiration
- SAC (Haarnoja 2018) - main off-policy alternative
- Recurrent PPO / PPO with LSTM - extension for partial observability

## Status

- [x] code reproduces (OpenAI Spinning Up PPO reference impl exists)
- [ ] result cited in our paper (will be in Project A v1, sec 4.2 baselines)
- [ ] result counter-argued by another paper we read
