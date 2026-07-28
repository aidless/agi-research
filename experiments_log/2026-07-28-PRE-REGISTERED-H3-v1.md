# Pre-registered H3 for Y1.3 v1.5 (longer PPO budget)

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md, test if longer PPO training
>         makes the Monitor signal useful. If H1 verdict was because
>         100K PPO was too short, longer training might let the
>         Monitor shape the policy meaningfully.

## 1. Background

H1 (LunarLander-v3, 100K PPO, n=15 per arm): H1 NOT supported.
  Real - Random = +13.6, t=0.78, NOT significant.
  Verdict: "Shaping helps; Monitor not validated."

H2 (Acrobot-v1, 100K PPO, n=5): Y1.3 ≈ PPO. Y1.3 doesn''t help.

**Question for H3**: was 100K PPO too short for the Monitor to be
useful? With 100K, PPO might not have fully converged, and the
Monitor signal may need more iterations to shape the policy.
500K (5x more) gives PPO much more time to learn AND to incorporate
the Monitor signal.

## 2. Hypothesis (PRE-REGISTERED)

**Env**: LunarLander-v3 (same as H1)

**H3**: With 500K PPO budget (5x more than H1''s 100K) and lambda=0.5,
  Y1.3 with trained SlotMonitor gives a higher mean return than Y1.3
  with random monitor, with delta > +10 AND Welch t > 2.0.

**H0 (null)**: Real and random give same mean return at 500K PPO
  (delta < +10 or t < 2.0).

**Decision rule**: Same as H1/H2.
  - If H3 supported (n=5+ per arm, Welch t > 2.0, delta > +10):
    "Y1.3 Monitor signal is informative at 500K PPO budget"
  - If H0 supported: "H1 verdict generalizes to 500K PPO: Monitor
    signal still not validated above shaping"

## 3. Pre-registered sample size

n=5 per arm (real vs random) at 500K PPO. Matches the original
Y1.3 sweep (5 seeds each).

NOTE: This is a pilot sample size, not n=10. If H3 is borderline,
I''ll consider extending to n=10. But I expect the H3 verdict
to be clear with n=5 (either Monitor helps at 500K or it doesn''t).

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - The PPO training crashes (Python exception)
  - The eval episodes are truncated
  - The seed number was set wrong (programmer error)

A seed is INCLUDED even if:
  - PPO fails to converge (PPO can fail; that's the data)
  - The Monitor fails to train (use whatever was trained)
  - The eval mean is very low

## 5. Pre-registered analysis plan

For each arm (real, random):
  - Compute per-seed mean eval return (50 episodes)
  - Compute aggregate mean, std
For pairwise comparison:
  - Welch t-test
  - Sign test
For the headline claim:
  - If Real > Random with t > 2.0 and delta > +10: claim "Y1.3
    Monitor signal is informative at 500K PPO"
  - Otherwise: claim "H1 verdict generalizes to 500K PPO: Monitor
    not validated"

## 6. Pre-registered stopping rule

Run to completion (n=5 per arm) without interim peeking. After
completion, report the verdict. Do NOT add seeds, drop seeds, or
re-run specific seeds based on results.

## 7. Deviation from prior protocol

This H3 sample size is n=5 (matching the original Y1.3 sweep),
not the n=10+ I used for H1 controls. The reason: 500K PPO is
5x more expensive than 100K. With n=10 per arm, the total cost
would be 100K seeds * 5x = expensive.

If H3 with n=5 shows a CLEAR verdict (delta > +20 with t > 2.0 or
delta < +5 with t < 1.0), I accept n=5. If H3 is borderline
(5 < delta < 20), I will report the verdict and note the n=5
limitation; I will not silently extend to n=10.

## 8. Pre-registration log

H3 was registered on 2026-07-28 BEFORE the 500K PPO sweep was
launched. Any change to the registered hypothesis, sample size,
decision rule, exclusion rules, analysis plan, or stopping rule
must be documented as a deviation and justified.
