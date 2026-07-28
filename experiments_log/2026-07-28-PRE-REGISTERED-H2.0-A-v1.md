# Pre-registered H2.0-A - Forward model + PPO (ICM-style exploration)

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md, test a NEW intervention
>         (forward model exploration bonus) instead of the failed
>         Monitor interventions. If forward model bonus helps, this
>         is a publishable positive result for curiosity-driven PPO.

## 1. Background

Y1.x sub-project (Sections 4.10.1-4.10.25): 4 H tests, 0 supported.
  - Monitor signal does NOT help PPO in any tested use case
  - Monitor architecture is real (AUROC 0.99) but ONLINE policy
    interventions do not work
  - Y1.x sub-project CLOSED

H2.0 is a NEW direction. Instead of using the Monitor, we use a
**forward model**: a small MLP that predicts next_state given
(current_state, action). The prediction ERROR is used as an
exploration bonus (high error = novel state = worth visiting).

This is the ICM (Pathak et al. 2017) / RND (Burda et al. 2018)
approach. It is a STANDARD curiosity-driven exploration method
that has been shown to help in many RL benchmarks.

H2.0-A hypothesis: forward model bonus helps PPO. Random forward
model (random weights) as control: should give less/no benefit
than trained forward model.

## 2. Hypothesis (PRE-REGISTERED)

**Env**: LunarLander-v3 (same as H1, H1.4)

**H2.0-A**: With 100K PPO budget and bonus_coeff=0.5, PPO with a
  TRAINED forward model (whose prediction error is added as
  exploration bonus) gives a higher mean return than PPO with a
  RANDOM-WEIGHT forward model (also added as bonus), with
  delta > +10 AND Welch t > 2.0.

**H0 (null)**: Trained and random forward model give same mean
  return (delta < +10 or t < 2.0).

**Decision rule**: Same as H1/H2/H3/H1.4.
  - If H2.0-A supported: "Forward model exploration bonus is
    informative above random"
  - If H0 supported: "Forward model exploration bonus does NOT
    help PPO either; the failed Y1.x verdict generalizes to
    forward model as well"

**Why H2.0-A is worth testing separately from Y1.x**:
- Different intervention class (forward model vs Monitor)
- Different theoretical basis (curiosity-driven vs failure
  prediction)
- Different signal source (state-prediction error vs
  failure probability)
- If H2.0-A supports, this is a publishable "ICM/RND works on
  LunarLander with PPO" finding

## 3. Pre-registered sample size

n=5 per arm (trained vs random). Matches the H3/H1.4 sample size.

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - PPO training crashes
  - Eval episodes truncated
  - Seed number set wrong

## 5. Pre-registered analysis plan

For each arm:
  - Per-seed mean eval return (50 episodes)
  - Aggregate mean, std
For comparison:
  - Welch t-test
For the verdict:
  - If Trained > Random with t > 2.0 and delta > +10: claim
    "Forward model exploration bonus is informative"
  - Otherwise: claim "H2.0-A NOT supported; forward model
    exploration bonus does not help PPO"

## 6. Pre-registered stopping rule

Run to completion (n=5 per arm) without interim peeking.

## 7. Pre-registration log

H2.0-A was registered on 2026-07-28 BEFORE the sweep was launched.
Any change to the registered hypothesis, sample size, decision
rule, or stopping rule must be documented as a deviation and
justified.
