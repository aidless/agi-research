# Decision Transformer (Chen et al. 2021)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- widely-cited, multiple reproductions
> One-line: Cast RL as sequence modelling. Treat trajectories as token
> sequences, train a transformer to autoregressively predict actions
> given states and (return-to-go) using standard cross-entropy/MSE loss.

---

## Problem

Off-policy RL methods (DQN, SAC) have replay buffers that grow
uncontrollably. Bootstrapping introduces bias. Reward design is
endlessly tricky. Policy gradient methods are high-variance.

What if we just throw away the RL frame and train a sequence model?
**Trajectories are sequences. Predict the next action.**

## Method

Trajectory representation:
- state s_t
- action a_t (continuous action; discretised)
- return-to-go R_t = sum_{tau=t}^{T} r_tau (cumulative return from t)

All three are concatenated into a token sequence:
```
(R_0, s_0, a_0, R_1, s_1, a_1, ..., R_T, s_T)
```

Model: a GPT-style transformer decoder with causal attention.

Training: standard next-element prediction with cross-entropy (or MSE
for continuous quantities).

Inference:
1. Specify desired return R_0 = R_target
2. Model predicts action a_0
3. Execute a_0
4. Observe (s_1, r_1)
5. Compute R_1 = R_target - cumulative_reward_so_far
6. Model predicts a_1 given (R_0, s_0, a_0, R_1, s_1) history
7. Repeat

## Empirical result

- Atari: comparable or better than strong baselines (DQN, C51) on
  offline RL benchmarks
- MuJoCo: comparable to or better than TD3 / SAC on offline RL
- Notable: matches SOTA without policy gradient, value bootstrapping,
  OR even off-policy correction.

## Criticisms

1. **Clunkier than policy gradient for online RL.** Sequence-model
   training is slow; online Decision Transformer variants exist but
   are non-trivial.

2. **Sensitive to dataset distribution**. Decision Transformer
   overfits to the training dataset's return distribution. At eval,
   you can ask for returns higher than seen in training, but it gets
   unreliable.

3. **No explicit credit assignment beyond context length**. If you have
   long-horizon tasks, you must keep a long history. Transformers
   handle context length poorly.

4. **"Return-to-go" is a leaky abstraction**. Setting R_target = some
   high value essentially gives the agent a goal. This is great for
   few-shot imitation but doesn't compose with sparse-reward envs.

5. **Doesn't actually outperform carefully-tuned PPO for online use**;
   its strength is offline / few-shot transfer.

## Connection to our program

Decision Transformer is **an alternative paradigm** to PPO-based RL.
For Project A, we chose PPO for clean decoupling story. But DT offers
a path to **return-conditioned self-monitoring**:

A Decision Transformer's predicted R_t is essentially a value estimate
for current trajectory with desired future return. We could place a
**monitor ABOVE Decision Transformer** that predicts catastrophic
failure based on the predicted action and trajectory history.

This is a **future Project A extension** beyond PPO. We will not
build it in Y0 but should note it as related.

More importantly, Decision Transformer's framing as "tokens all the way
down" is competitive with our Project D's "language-as-type-system"
vision. The DT approach says: every RL trajectory is just tokens; you
don't need RL specialised machinery. Our Project D approach says:
language is the type system over latent state. These are related but
distinct.

## Concrete next move

None for Y0. Possibly considered in Y2 if Project A extrapolates well.

## Confidence

HIGH for headline results.

Re-verify:
- exact numbers on Atari 50-game median
- the specific reward-conditioning behaviour

## Related papers

- Trajectory Transformer (Janner 2021): longer horizons, planning
- Gato (Reed 2022): DT-like but multi-task
- Online Decision Transformer variant (Zheng 2022)
- Decision Mamba (2024?): Mamba-based alternative

## Status

- [ ] mark as future Project A extension in v2 paper
- [x] cite in TASKBOOK architecture v2 (one line in planner block)
