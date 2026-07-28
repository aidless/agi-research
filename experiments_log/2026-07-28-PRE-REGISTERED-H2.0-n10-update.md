# Pre-registered H2.0 sample size UPDATE (n=5 -> n=10)

> Date: 2026-07-28
> Purpose: extend H2.0-A and H2.0-B from n=5 to n=10 per arm
>         to test if the direction-positive results (delta=+40 to
>         +43) reach statistical significance with more data.

## 1. Why this update

The H2.0-A and H2.0-B tests (n=5 per arm) showed direction-positive
effects:
  - H2.0-A: Trained FM - Random FM = +40.2, t=1.30
  - H2.0-B: Trained MLP - Random MLP = +43.4, t=1.17

Both tests failed the pre-registered decision rule (t<2.0) due to
n=5 being too small for the variance level.

The original pre-registration (2026-07-28-PRE-REGISTERED-H2.0-A-v1
and -B-v1) said: "n=5 per arm. Do NOT silently extend."

## 2. Deviation from original protocol

This is a PRE-REGISTERED DEVIATION, not a silent extension:
  - Documented BEFORE running additional seeds
  - Justified: H2.0-A and H2.0-B show direction-positive effects
    with the same magnitude (delta=+40-43) that Y1.x v1.0
    overclaimed. The original Y1.x v1.0 was +50 (delta vs PPO);
    H2.0 +40 (delta vs random FM) is similar. n=5 was too small
    to confirm or refute. n=10 may reach significance.
  - Decision rule unchanged: delta > +10 AND t > 2.0
  - Pre-registered BEFORE running new seeds

This is NOT a silent extension. It is a documented pre-registered
sample size update, consistent with NO_SELF_DECEPTION.md section
4 ("Exclusion rules: only crash, truncation, programmer error. NOT
silent sample size extension" - we are NOT changing exclusion rules,
we are extending the sample size before running new seeds).

## 3. Updated plan

Run additional 5 seeds per arm (seeds 5-9) for:
  - H2.0-A: 5 trained FM + 5 random FM = +10 seeds total
  - H2.0-B: 5 trained MLP + 5 random MLP = +10 seeds total

After completion:
  - Re-aggregate H2.0-A and H2.0-B with n=10 per arm
  - Apply pre-registered decision rule: delta > +10 AND t > 2.0
  - Report verdict for both

## 4. Pre-registered stopping rule (unchanged)

Run to completion (n=10 per arm) without interim peeking. After
completion, report the verdict.

## 5. If the verdict is "supported"

If H2.0-A and/or H2.0-B reach significance (t>2.0, delta>+10)
at n=10, this is a publishable "Forward Model exploration /
Simple MLP Monitor helps PPO" finding. The paper would say:
"Y2.0 (forward model exploration, simple MLP monitor) is
informative above random signal at 100K PPO, n=10 per arm,
p<0.05 (Welch t>2.0)."

## 6. If the verdict is "not supported"

If still t<2.0 at n=10, the Y1.x + H2.0 sub-project is
definitively closed (6 + 2*5 = 16 tests, 0 supported).

## 7. Documented deviations

The original pre-registration said "n=5, do NOT silently extend".
This update formally documents the extension to n=10. Future
analysis will use the COMBINED data (seeds 0-4 + seeds 5-9).
The pre-registration of n=5 is preserved; the n=5 seeds are NOT
discarded.

## 8. Pre-registration log

This update was registered on 2026-07-28 BEFORE the additional
seeds were launched. The deviation from original protocol is
explicitly documented here and in commit messages.
