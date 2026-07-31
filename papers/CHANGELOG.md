# Project G (H10) Changelog

A trail of every meaningful change to the H10 paper and associated scripts,
for reproducibility and reviewer accountability.

## v0.6.1 (2026-07-31) -- CURRENT DRAFT

Code & protocol changes:
- _fill_section_7_7.py hardened with sentinel-based replacement for §7.7.6
  verdict text; idempotent across re-runs.
- _fill_y5.py handles both template and already-filled states.
- _make_figures_v06.py reads baseline JSONs dynamically (no hardcoded n=5/20/100 numbers).
- All post-launch patches documented in `2026-07-31-h10-post-launch-patches.md`.

Pre-registration chain:
- Original pre-reg: `2026-07-28-PRE-REGISTERED-H10.md`.
- Amendment 1: `2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md` (n=20 GSM8K 200-tok).
- Addendum: `2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md`
  (kill switch tightened +0.05 -> +0.10 after power analysis).

Code patches in this paper:
- `real_llm_rollout_collector.py`:
    * CoT prompt for GSM8K (`prompt_style='gsm8k'`).
    * Last-20 token window (`tokens[-window:]` instead of `tokens[:window]`).
- `h10_real_pilot.py`:
    * Routes GSM8K mode to `gsm8k_local_loader` (datasets lib not installed).
    * Passes `prompt_style` to `collect_real_rollouts`.
    * Orphan LM-load block removed (technical debt cleanup).

## v0.6 (2026-07-31) -- initial GSM8K scaffolding

Added §7.7 "GSM8K 200-token follow-up (Pre-Reg Amendment 1)" as a TEMPLATE
section with placeholders for §7.7.4 / §7.7.5 / §7.7.6 / §7.7.8 / §7.8 / §7.9.
The template awaits real data to fill in from the aggregator.

## v0.5 (2026-07-29) -- stratified split

Added stratified train/eval split to fix v0.4's degenerate eval issue
(seed 2 of the deterministic-split n=5 had eval = all failures, AUROC
undefined). The stratified split ensures eval always has both classes by
splitting each class independently at 75/25.

Result: H10 REFUTED across two simple-arith samples (n=5 and n=20).

## v0.4 (2026-07-29) -- first n=5 pilot with deterministic split

Result: H10 REFUTED by direction (Joint > Frozen by 0.10, t=-0.516, NOT sig),
but the seed-2 eval set was degenerate (all-failures, AUROC undefined).

## (earlier)

Simple-arith n=100 was processed by the original `h10_real_pilot.py`
with `H10_MAX_NEW_TOKENS=64` and the old deterministic split that had
seed-level degeneracy. The n=100 result was the cleanest evidence for
the n=100 simple-arith row in the current table.

See `_h10_real_pilot_simple_*.log` and `_h10_n100_*.log` files in
`experiments_log/` for full per-seed detail.
