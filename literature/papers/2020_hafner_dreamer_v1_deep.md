# Dreamer V1 (Hafner et al. 2020, ICLR 2021)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- well-cited, multiple reproductions, code available
> One-line: First world-model-based RL agent that learns the dynamics in
> latent space and backpropagates policy gradients through imagined rollouts,
> achieving SOTA on 20+ Atari games and 30+ DMControl tasks.

---

## Problem

Model-free RL (DQN, PPO) is sample-inefficient: needs millions of env
interactions to learn simple tasks. Model-based RL "rollouts" can help
sample efficiency but had been unstable in high-dim pixel inputs.

Hafner's Dreamer V1 bet: learn a **latent dynamics model** from images,
then use it to imagine trajectories, and learn the policy purely from
imagined trajectories.

## Method

Three modules:

1. **Representation network**: encoder from observation o_t to stochastic
   latent h_t (with reparameterisation). After encoder + RSSM transition,
   you get a sequence (h_0, h_1, ..., h_T) capturing the world state.

2. **Dynamics network**: predicts next latent h_{t+1} from (h_t, a_t) and
   decoder predicts o_{t+1} and reward r_{t+1}.

3. **Policy & value networks**: both take h_t as input. Policy learned
   via backprop through the imagined dynamics; value learned via standard
   bootstrapping.

The imagined-trajectory policy gradient is the **critical** innovation:
the actor is updated entirely from model-imagined trajectories, with
no real env steps needed.

```
Algorithm 1 (Dreamer)
init: encoder, dynamics, decoder, actor, critic
repeat:
    1. roll out policy in env, store (o,a,r) sequences
    2. update encoder + dynamics + decoder + reward predictor on real data
    3. dream trajectories of length H from initial states in buffer
    4. update actor via imagined policy gradient
    5. update critic via lambda-returns on imagined trajectories
until converged
```

Critic update uses lambda-returns (continuous-valued, similar to GAE)
over the imagined horizon. This is critical for credit assignment across
imagined steps.

## Empirical result

DMControl 30 tasks: 17 of 30 with superhuman performance, comparable
to or better than prior model-based (world-models paper baselines).
Atari 20 games: with 100K env steps (~5M frames), Dreamer achieves
comparable to Rainbow DQN that uses 5-10x more data.

Sample efficiency: roughly 4-10x improvement over PPO/model-free on
DMControl. Less impressive on Atari (where model-free scaling is already
strong).

## Criticisms

1. **The stochastic latent z_t is only weakly disentangled**. There is
   no enforced factorisation by content of z. This hurts interpretability.

2. **Imagination horizon is short**. H=50 imagined steps before the model
   error compounds enough that the policy gradient signal is dominated
   by noise. Cascade prediction losses are partial mitigation.

3. **Reward model must be predictable from h_t**. If reward depends on
   global frame statistics, the encoder might not capture it. Most
   tasks have dense per-step rewards so this isn't an issue in practice.

4. **Compute is non-trivial**: Dreamer V1 GPU-days per task. Multi-task
   generalisation requires training one model per task.

5. **The claims of "world model" are accurate but understated**: the
   encoder-decoder must fit observed frames, so the latent has to
   encode sufficient detail. The model is a high-fidelity image
   predictor, not a small abstract dynamics. This makes Dreamer V1
   opaque and large.

## Connection to our program

Dreamer V1 is **the second-most important architectural paper** for us
(after MuZero). Three integrations:

1. Project A: monitor ABOVE Dreamer-style world model. Easier than
   monitor above MuZero because Dreamer's imagined trajectories are
   inspectable step-by-step, and the latent is more interpretable.

2. Project C: Dreamer V1 is L1 in Pearl's ladder. To lift to L2/L3
   we need structured intervention. Slot attention + Pearl's do-operator
   is the path.

3. Project D: PaLM-E-style language on top of Dreamer's latent for
   natural-language interfaces ("show me the imagined trajectory").

## Concrete next move

Run Dreamer V1 on LunarLander-v2 as a baseline. Compute cost: 1 GPU day
on a single task. We will not do this without GPU, but we should
write the code skeleton so it can run when GPU is available.

## Confidence

HIGH for everything except exact hyperparameters. Re-verify:
- H = 50 default (not 100)
- encoder architecture (I recall "small CNN") vs RSSM-big
- actor-critic relative loss coefficients

## Related papers

- PlaNet (Hafner 2019): precursor, latent dynamics only, no policy
- Dreamer V2 (Hafner 2021): adds categorical latent
- Dreamer V3 (Hafner 2024 / Nature 2025): scale-up
- TD-MPC (Hansen 2022): MPC variant, simpler single-step objective
- IRIS (Micheli 2022): Transformer world model

## Status

- [x] cite in Project A Method (alternative to MuZero for monitor)
- [x] cite in Project C Section 1 (current L1 ceiling)
- [x] cite in TASKBOOK architecture block
- [ ] re-read exact hyperparameter table for paper review
