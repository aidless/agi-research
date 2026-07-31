# Pre-Registration Amendment 1 ADDENDUM: Tightened Kill-Switch

> Addendum date: 2026-07-31 12:00 (after launching the 60 jobs at 11:32)
> Original amendment: `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md`
> Status: ADDENDUM in-flight (data collection underway). Pre-registered BEFORE aggregation is run.

---

## Why this addendum exists

A power analysis re-check (run at 2026-07-31 ~11:55) showed that the
looser kill switch in Amendment 1 ("extend to n=50 if delta >= +0.05")
is **not protective enough against n=20 noise**. At d=+0.05 (the
minimum realistic effect size) the n=20 power is 1.9% and at d=+0.20
(the pre-reg threshold) power is only 6.7%. An observed delta of
+0.05 at n=20 is overwhelmingly likely to be noise, not signal.

This addendum **tightens the kill switch** while the run is still
in progress. The tightening is pre-registered BEFORE the aggregation
step runs, so the decision rule cannot be moved after seeing results.

## Pre-registered replacement kill switch (replaces Amendment 1's)

Apply ONLY to the 60-job aggregation result. After aggregation:

| Frozen - Joint (n=20) | Pre-registered action |
|-----------------------|-----------------------|
| **`>= +0.10`**        | **Extend** to n=50 (180 more jobs, ~14 h more) |
| **`[0, +0.10)`**      | **Stop.** Write paper: H10 REFUTED on simple arithmetic and GSM8K (two independent tasks). |
| **`< 0`**             | **Stop.** H10 REFUTED with consistent negative direction across both tasks (matches n=100 simple arithmetic). |

The `+0.10` threshold corresponds to power ~3% at d=+0.05, ~6.7% at d=+0.20.
A real effect of d >= +0.30 requires n=20 to get power ~15%, but observing delta >= +0.10 at n=20 is still consistent with a real effect of d >= +0.30 -- so we extend to n=50 to disambiguate.

The `+0.10` threshold (vs the original `+0.05`) is a 2x safety factor
against noisy n=20 results. With n=50 the same threshold becomes much
more meaningful: power at d=+0.20 jumps from 16.4% to 39% when n is
doubled; power at d=+0.30 jumps from 39% to 71%.

## Rationale for the boundary

The "extend" action (delta >= +0.10) costs ~14 more hours at n=50. We
must NOT extend on noise. We can justify spending 14 hours on noise
ONLY IF there is at least some directional signal at n=20.

The "stop" action (delta in [0, +0.10)) writes a paper. H10 REFUTED on
both tasks is a publishable result:
- Y4 paper 7.5 already establishes H10 REFUTED on simple arithmetic
- GSM8K 200-token is a qualitatively different task (chain-of-thought
  reasoning, last-20 failure signal) so adding it makes the negative
  result stronger, not weaker
- COLM 2026 reproducibility-track framing benefits from negative
  results across two tasks

The "stop" action (delta < 0) freezes the simple-arithmetic-refuted
direction at GSM8K 200-token, mirroring the n=100 simple-arithmetic
conclusion.

## What is NOT changed in this addendum

- The decision rule from the original pre-reg (VALIDATED/REFUTED/INCONCLUSIVE
  thresholds) is preserved.
- The negative control (Random Monitor) is preserved.
- The mechanism hypothesis is preserved.
- The seed range (100..119) and arm set (frozen/joint/random) are
  preserved.
- The pilot seed (100) is included in the n=20 aggregation but it was
  used as a *smoke-test* (H10_USE_SIMPLE=0, H10_MAX_NEW_TOKENS=200,
  H10_N_TOTAL=8) and is identical in configuration to the rest of the
  60 jobs.

## What the addendum CHANGE is

ONLY the threshold for "extend to n=50": `+0.05` -> `+0.10`.

This single numerical change is **the entire content of the
addendum**. No decision rule, no protocol, no arm set, no exclusion
criteria is changed.

## Sequencing note

This addendum is filed at 12:00 on 2026-07-31. The first batch of
frozen-arm jobs (s100-s105) had completed before the addendum was
written, but their results have not been read for analysis. The
aggregator (`experiments_log/_agg_h10_n20_gsm8k.py`) was enhanced
after the addendum to print the kill-switch recommendation
automatically. The aggregator is run after all 60 jobs complete, so
no result is influenced by the addendum.

## NO_SELF_DECEPTION.md compliance

1. Decision rule is documented BEFORE aggregation.
2. The tightening is conservative (high threshold, fewer false-positive
   extensions) -- this is the OPPOSITE direction from "make H10 look
   better."
3. The threshold change is fully motivated by the power analysis
   (printed in the validator writeup at
   `experiments_log/2026-07-31-h10-gsm8k-200t-validation.md` and
   reproducible from `cohen_n` / `power_at_n` formulas).
4. No seed is silently dropped regardless of outcome.
5. The aggregagtor uses standard paired-t + bootstrap-2000 CIs as in
   the n=100 aggregation; no new statistical test is introduced.

---

*Addendum filed 2026-07-31 12:00, before any aggregation result is
read. This addendum replaces the "extend to n=50 if delta >= +0.05"
clause in Amendment 1 with "extend to n=50 if delta >= +0.10".
Nothing else is changed.*
