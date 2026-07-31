# Pre-Registration Amendment 1: H10 GSM8K 200-token Extension

> Amendment date: 2026-07-31
> Original pre-registration: `experiments_log/2026-07-28-PRE-REGISTERED-H10.md`
> Author: Liu Zewen (with Codex agent)
> Status: AMENDMENT to a pre-registered hypothesis; no override of the decision rule.

---

## What this amendment changes

The original H10 pre-registration specified GSM8K-style math reasoning
on a small LM but the running implementation (`h10_real_pilot.py`)
defaulted to a simpler arithmetic dataset (64-token max) for the
**CPU pilot** stage. Three sample sizes (n=5, n=20, n=100) on the
simpler arithmetic 64-token task produced **`H10 = REFUTED`** in every
sample size, with AUROCs at the chance level (~0.5). See
`2026-07-29-H10-stratified-n5-result.md`,
`2026-07-30-_h10_n100_agg.log` and the n=100 bootstrap JSON.

This amendment **does NOT change the H10 decision rule**. It extends
the experimental surface in two dimensions so that the H10 hypothesis
is tested under conditions where the failure-prediction signal is
stronger (longer traces, real chain-of-thought reasoning):

1. **Sample size: n=20 seeds** (3 arms x 20 seeds = 60 jobs).
   - This is the pre-registered extension point in the original
     pre-reg ("extension to n=15 ... and again to n=20 if needed").
2. **Token budget: H10_MAX_NEW_TOKENS = 200** (up from 64).
   - GSM8K chain-of-thought traces for Qwen2.5-1.5B typically run
     100-250 tokens, so 200 covers the realistic CoT length without
     truncation in the modal trace.

All other parameters (LM = Qwen2.5-1.5B-Instruct, Monitor =
LLMSlotMonitor window=20 slot_dim=32 n_slots=4, architectures =
3 arms Frozen / Joint / Random, decision rule, success criterion,
refutation criterion, exclusions, negative control) are unchanged.

## Why this amendment is needed (and why it is "back to the original")

The original H10 pre-registration named GSM8K as the primary task
("GSM8K-style math reasoning traces from a frozen small LM"). The
current H10 implementation defaulted to a synthetic simple-arithmetic
substitute because the pilot ran on CPU without the `datasets`
library. n=100 of that substitute established that the failure signal
in 64-token simple-arithmetic traces is too weak to discriminate
between any of the three arms (all three within +/-0.02 of chance).

This amendment **re-aligns the running implementation with the
original pre-registered task** by:

- Switching the dataset from simple arithmetic to GSM8K
  (using the local arrow loader at
  `F:\hf_cache\datasets\openai___gsm8k\...\gsm8k-test.arrow`).
- Lengthening the trace budget to 200 tokens so the chain-of-thought
  reasoning has room to develop.

This is therefore not a `new hypothesis` but a `faithful execution`
of the original H10 pre-registration on a sample size (n=20) and
token budget (200) that the original pre-reg's extension protocol
explicitly anticipates.

## What stays the same

- **Decision rule** (from the original pre-reg):
  - VALIDATED iff Frozen > Joint by >0.05 AND Welch t > 2.0 on n=5+
    seeds AND Frozen > Random by >0.10.
  - REFUTED iff Frozen <= Joint OR Frozen == Random.
  - INCONCLUSIVE iff Frozen > Joint by >0.05 but t < 2.0.
- **Three arms**: Frozen, Joint, Random (negative control).
- **No silent exclusions**: every seed's result is reported.

## New specifics for this amendment

| Parameter | Value | Source |
|-----------|-------|--------|
| n_seeds_per_arm | 20 | Amendment 1 (was 5 in original pre-reg) |
| total_jobs | 60 | 3 arms x 20 seeds |
| dataset | GSM8K test set | Original pre-reg (re-aligned) |
| dataset_loader | gsm8k_local_loader.load_gsm8k_local | New (bypasses missing `datasets` lib) |
| LM | Qwen/Qwen2.5-1.5B-Instruct (CPU, fp16) | Original pre-reg |
| max_new_tokens | 200 | Amendment 1 (was 64 in CPU pilot) |
| monitor_window | 20 | LLMSlotMonitor default |
| feature_last_n | 20 (last 20 tokens, NOT first 20) | Amendment 1 (fix bug noted in pilot reviews) |
| prompt | "Question: ...\nLet's think step by step.\n" | Amendment 1 (CoT format required for word problems) |
| seed range | 100..119 | Same range as n=20 simple-arith |
| compute_estimate | 60 jobs x ~5.4 min = ~5.4 h | Observed single-job rate (n=100 took 1.77 min on simple arith; 200-token GSM8K 3-5x slower) |

## Pre-registered decision actions (kill switch)

After the 60 jobs finish aggregating, the following pre-registered
actions apply:

1. **Frozen - Joint >= +0.05** (any direction consistent at n=20):
   - The pre-reg extension protocol triggers:
     extend to n=50 (180 jobs) for a power-adequate test.
2. **Frozen - Joint in [0, +0.05]**:
   - Stop. Write the GSM8K + simple-arithmetic + slot-Monitor
     evidence chain. Report H10 REFUTED on both tasks.
3. **Frozen - Joint < 0** (Joint > Frozen):
   - Stop. Mirror the n=5/20/100 simple-arithmetic result.
     Report H10 REFUTED with a strong negative direction across
     two independent tasks.

In all three cases, the **decision rule is preserved**, the
**kill switch is recorded before data collection**, and
**no seed is silently dropped**.

## Cross-references

- Original pre-reg: `experiments_log/2026-07-28-PRE-REGISTERED-H10.md`
- n=100 simple-arith aggregation:
  `experiments_log/_h10_n100_agg.log`,
  `experiments_log/_h10_n100_bootstrap.json`
- n=20 simple-arith aggregation: `experiments_log/_h10_n20_bootstrap.json`
- n=5 stratified result: `experiments_log/2026-07-29-H10-stratified-n5-result.md`
- New pilot output (this amendment): to be logged under
  `experiments_log/_h10_n20_gsm8k_<arm>_s<seed>.log`
  per-arm, 60 jobs total.

## NO_SELF_DECEPTION.md compliance

1. The decision rule is preserved (not changed by this amendment).
2. The negative control (Random Monitor) is preserved.
3. The mechanism hypothesis (joint Monitor gradient dragged by LLM
   update) is preserved.
4. The replication plan (n=20 -> n=50 if direction-positive but
   underpowered) is preserved and pre-committed.
5. Boundary on what is NOT claimed: the amendment does NOT claim
   that GSM8K 200-token will give a positive result; it only
   **widens the experimental surface** so the failure signal can
   be detected if it exists.
6. The amendment is **documented BEFORE data collection** and
   added to the experiments log next to the original pre-reg.

---

*Amendment 1 filed 2026-07-31, before any n=20 GSM8K 200-token
data is collected. No further amendments are pre-committed; any
additional change will be filed as Amendment 2.*
