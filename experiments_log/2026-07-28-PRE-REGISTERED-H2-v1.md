# Pre-registered H2 for Y1.3 v1.4 (cross-env replication)

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md, replicate the H1 3-way control
>         on a different env. If H2 supports (Monitor signal helps
>         in some envs), then H1''s verdict is env-specific. If H2
>         fails, then H1''s verdict generalizes across envs.

## 1. Background

H1 (LunarLander-v3, n=15 per arm): H1 NOT supported. Real Monitor
signal does not significantly improve over random shaping (delta=+13.6,
t=0.78). v1.3 verdict: "shaping helps; Monitor not validated".

H2 (this): does the same protocol on a different env show a
different verdict? Two possible outcomes:
  (a) H2 supported (Real > Random, t > 2.0, delta > +10):
      H1 verdict is env-specific; Monitor helps in some envs
      (just not LunarLander)
  (b) H2 NOT supported (Real ~ Random, t < 2.0):
      H1 verdict generalizes; the Monitor architecture does not
      transfer to policy improvement at this PPO budget

## 2. Hypothesis (PRE-REGISTERED, stated BEFORE running H2)

**Env**: Acrobot-v1 (a different env from H1)

  Acrobot has:
  - Discrete actions (3)
  - Low-dim obs (6)
  - Sparse-ish reward (range -500 to 0)
  - PPO at 100K typically converges to ~-87 (near -80 lower bound)
  - Different failure modes than LunarLander

**H2**: On Acrobot-v1 with 100K PPO budget and lambda=0.5,
  Y1.3 with a trained SlotMonitor gives a higher mean return than
  Y1.3 with a random monitor of the same signal magnitude, with
  effect size at least +10 (delta_mean) and Welch t > 2.0.

**H0 (null)**: Y1.3 real and Y1.3 random give the same mean return
  on Acrobot (delta < +10 or t < 2.0).

**Decision rule**: Same as H1:
  - If H2 supported (n=10+ per arm, Welch t > 2.0, delta > +10):
    "Y1.3 Monitor signal is informative on Acrobot"
  - If H0 supported: "H1 verdict generalizes: shaping helps but
    Monitor does not significantly transfer to policy improvement"

## 3. Pre-registered sample size

n=10 per arm (real vs random) on Acrobot. With n=10 per arm, Welch
t-test with d=0.6 and alpha=0.05 has ~70% power. Matches the n
used for H1 NC and inverse controls.

(Note: I''m using n=10 here vs n=15 for H1 real because:
  - H1 real is already at n=15 on LunarLander; extending Acrobot
    real to n=15 doubles compute without clear benefit
  - n=10 is the minimum for the pre-registered decision rule
  - If H2 needs n=15 for stronger conclusion, I''ll do it later)

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - The PPO training crashes (Python exception)
  - The eval episodes are truncated (env step limit hit)
  - The seed number was set wrong (programmer error)

A seed is INCLUDED even if:
  - PPO fails to converge
  - The Monitor fails to train
  - The eval mean is very negative (failure modes)

## 5. Pre-registered analysis plan

For each arm (real, random):
  - Compute per-seed mean eval return (50 episodes)
  - Compute aggregate mean, std
For pairwise comparison:
  - Welch t-test
  - Bootstrap 95% CI on delta (10K resamples)
  - Sign test
For the headline claim:
  - If Real > Random with t > 2.0 and delta > +10: claim "Y1.3
    Monitor signal is informative on Acrobot"
  - Otherwise: claim "H1 verdict generalizes: shaping helps; Monitor
    does not specifically help"

## 6. Pre-registered stopping rule

We run the experiment to completion (n=10 per arm) without interim
peeking. After completion, we report the verdict per the decision
rule. We do NOT add seeds, drop seeds, or re-run specific seeds
based on the results.

## 7. Pre-registration log

H2 was registered on 2026-07-28 BEFORE the Acrobot sweep was
launched. Any change to the registered hypothesis, sample size,
decision rule, exclusion rules, analysis plan, or stopping rule
must be documented as a deviation and justified.

Deviations are tracked in the experiment log, not silently applied.
