# Distributional RL - C51 (Bellemare 2017) / QR-DQN (Dabney 2018) / IQN (Dabner 2018)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Instead of learning a single Q-value for each (s,a), learn a
> distribution over Q values. The expected value of the distribution is
> the Q estimate; the variance carries information about policy risk.

---

## Problem

Q-learning learns a single point-estimate of expected return: Q(s, a)
= E[r_t + gamma*r_{t+1} + ...]. But returns are stochastic; the
_distribution_ over returns tells you more:
- variance of return (risk)
- quantiles (for risk-aware policies)
- multimodal distributions (e.g., polysemy in envs)

Distributional RL: learn the entire return distribution.

## Method

C51 (Categorical 51-atom DQN):
- parameterise a categorical distribution over 51 "atoms" of returns
- use Bellman update on the **distribution**, not the expectation
- loss = cross-entropy between predicted and target distributions

QR-DQN (Quantile Regression DQN):
- learn a fixed number of quantiles (e.g., 200)
- loss = quantile regression (Huber-quantile loss)

IQN (Implicit Quantile Network):
- learn the quantile values themselves via implicit quantile
  regression; more flexible than fixed-quantile approaches

All three leverage the core insight: the distribution, not the mean,
is what RL should reason about.

## Empirical result

- Median human-normalised score on Atari 57 games improves significantly:
  - DQN: ~80%
  - C51: ~150%
  - QR-DQN: ~190%
  - IQN: ~230%

Distributional methods outperform vanilla DQN without architectural
changes to the network; just changing how Q is parameterised.

## Criticisms

1. **Hyperparameters multiply**. Number of atoms, quantile count, etc.
   add knobs.

2. **Risk-aware policies require care**. Expected value of a risky
   distribution may not reflect human preference for safety.

3. **Computational overhead**. Tracking a distribution is more
   memory/compute than tracking a single value.

4. **Theoretical benefit is well-established but practical benefit
   varies**. On some games, distributional RL doesn't help much.

## Connection to our program

Distributional RL has potential connections to Project A and Project C:

1. **Project A monitor input**: a distributional Q value carries more
   information than a scalar Q. Our Monitor's input features could
   include the variance of Q (risk estimate), not just the mean.

2. **Project C (causal)**: risk-aware planning. If our world model can
   reason about variance of next-state, it can flag when an action
   leads to high-variance outcomes (= likely catastrophic failure).
   This is a different monitor signal than success prediction.

3. **Beyond**: in Dreamer V3, distributional value heads are used.
   Dreamer V3's robustness may partly be due to distributional value.

## Related papers

- C51 (Bellemare 2017) - the origin
- QR-DQN (Dabney 2018)
- IQN (Dabney 2018)
- FQF (Yang 2019) - further quantile refinement
- Dreamer V3 distributional critic

## Status

- [x] cite in Project A monitor input feature design
- [ ] consider variance-of-Q feature in monitor (low-priority follow-up)
