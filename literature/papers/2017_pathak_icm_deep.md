# Curiosity / ICM (Pathak et al. 2017)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: Add an intrinsic reward signal based on how poorly the
> agent's dynamics model predicts the next state. Cannot get stuck
> because the agent's own ignorance drives it to explore new things.

---

## Problem

Sparse-reward RL struggles because the agent never sees positive reward
during exploration. Solution historically: hand-engineer shaped rewards.
Better solution: a self-supervised curiosity signal.

## Method

Two networks in addition to the RL policy:

1. **Inverse dynamics model**: predicts the action that was taken given
   (s, s'). This learns features that are action-relevant -- they
   predict what was done.

2. **Forward dynamics model**: predicts s' given (s, a) using the features
   from (1).

The **intrinsic reward** at each step is the prediction error of the
forward model.

```
r_intrinsic = || f(s,a) - s' ||^2
```

If the agent's forward model fails to predict the next state, the
agent "wants" to go there (because that's where uncertainty is).
The agent is intrinsically rewarded by exploring.

In practice: features from inverse dynamics are used (not raw states)
because raw-pixel prediction error is dominated by irrelevant variance
(visual flicker).

## Empirical result

ICM + A3C on Montezuma's Revenge (the famous hard-exploration Atari game):
- solves the first room (reward ~100)
- vanilla A3C with sparse reward: 0

Subsequent work (RND, AMIGo, etc.) extended this to harder problems
including Pitfall, Gravitar.

## Criticisms

1. **Noisy-TV problem**. A TV showing random content has high
   prediction error forever. Agent gets "stuck" watching TV. Multiple
   mitigations proposed (stochastic features, model ensembles, etc.)

2. **Feature design matters**. Inverse dynamics features aren't always
   right; sometimes you want features that ARE action-relevant.

3. **Magnitude mismatch**. The intrinsic reward can dwarf/extrinsic
   reward across training scales. Calibration is needed.

4. **Sample inefficiency is unchanged**. Curiosity helps exploration but
   doesn't make the underlying RL faster.

## Connection to our program

ICM has indirect connections:

1. **For Project B (cross-domain)**: Curiosity-style "novelty" of new
   environment drives exploration. This composes with our H2 transfer
   claim: a curiosity-driven agent is more likely to attempt novel
   actions in a new env.

2. **For Project A monitor design**: We can define "failure" partially
   in terms of forward-model-error spikes. If the monitor's prediction
   error spikes, the policy may be doing something novel and likely-
   to-fail. We could add this as a feature in our BCE monitor input.

3. **For Project C (causal)**: curiosity-style prediction error on
   per-slot transitions may reveal which slots are causally "active".
   Slots with high prediction error are the ones being intervened on.

## Related papers

- RND (Burda 2018): random network distillation, similar spirit
- AMIGo (Campero 2021)
- Go-Explore (Ecoffet 2021): different exploration paradigm
- Feudal Networks (Vezhnevets 2017): hierarchical exploration

## Status

- [x] cite in Project B Section 2 (related exploration work)
- [ ] possible follow-up feature for Project A monitor input
