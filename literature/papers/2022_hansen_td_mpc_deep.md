# TD-MPC (Hansen et al. 2022)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Combine latent dynamics + terminal-value function + MPPI-like
> sampling. Single latent forward model, sampled trajectory rollouts,
> pick best by value estimate.

---

## Problem

Dreamer V1/V2 expensive to train. MuZero expensive (MCTS at every step).

TD-MPC bet: a simple latent dynamics model + sampled rollouts +
value terminal cost is enough to learn a useful policy, with one tenth
of the compute.

## Method

Five components:
1. Latent encoder: maps obs to latent z
2. Latent dynamics: predicts next latent given action
3. Reward predictor: latent -> reward
4. Value function: latent -> V(s)
5. Policy: latent -> action

At every env step:
- Sample K candidate trajectories of length H (H=5 typically) from
  current state via the learned dynamics.
- For each candidate, compute (sum of predicted rewards) +
  terminal V(end).
- The action of the best candidate's first step is taken.

This is **MPPI-like sampling**, lighter than full MCTS.

Training:
- encoder, dynamics, reward all trained via standard predictive losses
- value trained via TD(lambda) on actual env trajectories
- policy trained via backprop through the latent dynamics + value

## Empirical result

- DMControl 100 tasks: better or comparable to Dreamer at ~5x less
  compute
- Sample efficiency: 100K env steps reaches performance that Dreamer
  reaches at 500K
- Some hard locomotion tasks where Dreamer struggles, TD-MPC solves

## Criticisms

1. **Sampled trajectories cover only local horizon**. Without MCTS, the
   policy may be too short-sighted for long-horizon tasks.

2. **Reward predictor accuracy limits**. If reward is hard to predict
   from latent (e.g., sparse rewards), the MPPI rollouts fail.

3. **Single-sample trajectories are noisy**. K samples mitigate but
   not eliminate.

4. **Code release quality was initially uneven**; replication
   required some calibration.

## Connection to our program

TD-MPC is **a strong candidate Project A experiment environment
alternative**:
- cleaner pipeline than Dreamer V1
- direct value-function access (which is what Monitor predicts!)
- means we can build a "decoupled monitor above TD-MPC" that has
  direct access to Q values, value estimates, predicted rewards -- a
  rich state for failure prediction

Specifically: TD-MPC's "prediction-error spike" is a strong intrinsic
monitor signal. The Monitor can be a thin MLP that takes
(latent, predicted_reward, predicted_value, predicted_next_state_error)
as features and outputs failure probability.

This is a future Y1-Y2 extension of Project A.

## Related papers

- Dreamer V1/V2/V3
- MuZero / EfficientZero
- TD3 (Fujimoto 2018): not a world model but provides framework
- IRIS (Micheli 2023): transformer world model for Atari
- Dreamer V3 (Hafner 2025) - supersedes Dreamer V2 in performance

## Status

- [x] cite in Project A future work extension
- [ ] mark as Y2 candidate for "monitor above TD-MPC"
