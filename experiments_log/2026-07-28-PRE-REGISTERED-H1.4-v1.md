# Pre-registered H1.4 - Monitor as exploration bonus

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md, test a DIFFERENT use case of
>         the Monitor. Y1.3 used Monitor as reward shaping (subtract).
>         H1.4 uses Monitor as EXPLORATION BONUS (add). This is a
>         different intervention, not a parameter tweak.

## 1. Background

Y1.3 (v0.0-v1.5): Monitor used as REWARD SHAPING.
  shaped_reward = env_reward - lambda * Monitor_prob(window)
  H1 (100K PPO): NOT supported. Real - Random = +13.6, t=0.78.
  H2 prelim (Acrobot): Y1.3 ~ PPO.
  H3 (500K PPO): NOT supported. Real - Random = -53.1, t=-2.16.
  **Y1.3 sub-project CLOSED.**

Y1.3 is reward shaping (penalize risky states). This made the
Monitor into a "noise" signal at long PPO training (H3).

H1.4 tests a different use case: **exploration bonus** (reward
visiting risky/uncertain states). Conceptually similar to:
  - RND (Burda et al. 2018): reward visiting states the predictor
    is uncertain about
  - ICM (Pathak et al. 2017): reward visiting states with high
    forward-model error
  - MaxEnt exploration (Hazan et al. 2019): reward high-entropy states

For the Monitor specifically: high failure probability = "states
the agent fails in" = states worth exploring (so the agent learns
how to handle them).

H1.4 use case:
  exploration_bonus = beta * Monitor_prob(window)
  shaped_reward = env_reward + exploration_bonus
  (note: PLUS, not minus)

The sign is reversed from Y1.3. Y1.3 said "stay away from risky
states". H1.4 says "explore risky states (so you learn to handle
them)".

## 2. Hypothesis (PRE-REGISTERED)

**Env**: LunarLander-v3 (same as H1, H3)

**H1.4**: With 100K PPO budget and beta=0.5, Y1.4 with trained
  SlotMonitor used as exploration bonus gives a higher mean return
  than Y1.4 with random monitor (also as exploration bonus), with
  delta > +10 AND Welch t > 2.0.

**H0 (null)**: Y1.4 real and Y1.4 random give same mean return
  (delta < +10 or t < 2.0).

**Decision rule**: Same as H1/H2/H3.
  - If H1.4 supported: "Monitor as exploration bonus is informative
    above random"
  - If H0 supported: "Monitor signal is not useful as exploration
    bonus either; the Monitor architecture provides no policy
    benefit at this PPO budget"

**Why H1.4 is worth testing separately**:
- Different intervention (PLUS vs MINUS) than Y1.3
- Different theoretical basis (curiosity vs reward shaping)
- Different effect expected: exploration bonus REWARDS the agent
  for visiting risky states, which is the OPPOSITE of reward
  shaping
- If H1.4 also fails, we have strong evidence that the Monitor
  signal is not useful for policy improvement in any form

## 3. Pre-registered sample size

n=5 per arm (real vs random) for the headline comparison. Matches
the H1 sweep sample size and the H3 sample size.

NOTE: H1.4 is a "different intervention" test, not a "replicate
Y1.3 with same n" test. n=5 is sufficient to detect large
effects (H3 caught -53 with n=5). If borderline, I will report
the verdict and note the n=5 limitation; I will not silently
extend to n=10.

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - The PPO training crashes (Python exception)
  - The eval episodes are truncated
  - The seed number was set wrong (programmer error)

## 5. Pre-registered analysis plan

For each arm (real, random):
  - Compute per-seed mean eval return (50 episodes)
  - Aggregate mean, std
For pairwise comparison:
  - Welch t-test
  - Sign test
For the headline claim:
  - If Real > Random with t > 2.0 and delta > +10: claim "Monitor
    as exploration bonus is informative above random"
  - Otherwise: claim "H1.4 NOT supported; Monitor is not useful
    as exploration bonus either"

## 6. Pre-registered stopping rule

Run to completion (n=5 per arm) without interim peeking.

## 7. Comparison to Y1.3 (H1.4 vs Y1.3)

If H1.4 supports (Monitor useful as exploration), this would
suggest the failure mode of Y1.3 was the SIGN of the use case
(reward shaping penalizes Monitor signal, but exploration bonus
rewards it). If H1.4 also fails, the Monitor architecture is
simply not useful for policy improvement at this budget.

## 8. Pre-registration log

H1.4 was registered on 2026-07-28 BEFORE the H1.4 sweep was
launched. Any change to the registered hypothesis, sample size,
decision rule, or stopping rule must be documented as a deviation
and justified.
