# Pre-registered H1 for Y1.3 v1.3 (next iteration)

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md section 1.4, every future
>         experiment must have a pre-registered hypothesis.
> This is the pre-registration for the v1.3 iteration of Y1.3.

## 1. Background (what we know from v1.0, v1.1, v1.2)

v1.0 (ef90c2c): claimed "Y1.3 = +50 over PPO, t=6.76" - RETRACTED
v1.1 (e515565): NC showed random monitor gives +58, refuted v1.0
v1.2 (8faf30b): 3-way control shows real > random~inverse by +22-25
  (n=5 each for random/inverse, n=15 for real; not stat-sig)

The +22-25 Monitor signal delta above non-informative shaping is
the candidate publishable finding. But it is not statistically
significant with current n.

## 2. Hypothesis (PRE-REGISTERED, stated BEFORE running v1.3)

**H1**: On LunarLander-v3 with 100K PPO budget and lambda=0.5,
  Y1.3 with a trained SlotMonitor gives a higher mean return than
  Y1.3 with a random monitor of the same signal magnitude, with
  effect size at least +10 (delta_mean) and Welch t > 2.0.

**H0 (null)**: Y1.3 real and Y1.3 random give the same mean return
  (delta < +10 or t < 2.0).

**Decision rule**: If H1 is supported (n=10+ per arm, Welch t > 2.0,
  delta > +10), we have publishable evidence that the Monitor
  signal is informative above shaping. If H0 is supported, Y1.3
  is reframed as "reward shaping helps regardless of signal source"
  (v1.1 conclusion) and the Monitor architecture is not specifically
  validated.

## 3. Pre-registered sample size

Per H1 decision rule: n=10+ per arm. With n=10 per arm, Welch
t-test with d=0.6 and alpha=0.05 has ~70% power. We will run
n=15 per arm to reach ~85% power (matches v1.2's real-monitor n).

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - The PPO training crashes (Python exception, not convergence failure)
  - The eval episodes are truncated (env step limit hit)
  - The seed number was set wrong (programmer error)

A seed is INCLUDED even if:
  - PPO fails to converge (PPO can fail; that's the data)
  - The Monitor fails to train (use whatever was trained)
  - The eval mean is very low (negative, etc.)

## 5. Pre-registered analysis plan

For each arm (real, random, inverse):
  - Compute per-seed mean eval return (50 episodes)
  - Compute aggregate mean, std, t-stat
For pairwise comparisons:
  - Welch t-test (unequal variance)
  - Bootstrap 95% CI on delta (10K resamples)
  - Sign test (proportion of seeds with positive delta)
For the headline claim:
  - If real > random with Welch t > 2.0 and delta > +10: claim "Y1.3
    Monitor signal is informative above shaping"
  - Otherwise: claim "Y1.3 helps via reward shaping; Monitor signal
    effect not established"

## 6. Pre-registered stopping rule

We will run the experiment to completion (n=15 per arm) without
interim peeking. After completion, we report the verdict per the
decision rule. We will NOT add seeds, drop seeds, or re-run
specific seeds based on the results. If the verdict is "H1
not supported", we report that as-is.

## 7. Pre-registration log

This H1 was registered on 2026-07-28 BEFORE the v1.3 sweep was
launched. Any change to the registered hypothesis, sample size,
decision rule, exclusion rules, analysis plan, or stopping rule
must be documented as a deviation and justified.

Deviations are tracked in the experiment log, not silently applied.
