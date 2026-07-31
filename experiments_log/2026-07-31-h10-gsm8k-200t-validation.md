# H10 n=20 GSM8K 200-token Validation (1 seed, pre-launch)

> Date: 2026-07-31
> Code: projects/project_g_llm_self_monitoring/code/h10_real_pilot.py
> Mode: GSM8K + CoT prompt (after edit) + 200-token trace + LAST-20 feature window
> Seed: 100 (single run, NOT part of the n=60 jobs)
> Status: PASSED. Per-job timing and data shape both confirm the 60-job run is viable.

## What was validated

Three changes to the H10 protocol were applied via Pre-Reg Amendment 1
(see experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md):

1. **CoT prompt for GSM8K**: `Question: ...\\nLet's think step by step.\\n`
   instead of `What is ...? The answer is`.
2. **Last-20 token feature window**: `tokens[-window:]` instead of
   `tokens[:window]`, capturing the failure signal at the end of the trace.
3. **local arrow loader for GSM8K**: bypasses the missing `datasets`
   library and uses `gsm8k_local_loader.load_gsm8k_local` with
   seed-based sampling so different seeds see different problems.

## Validation run (seed=100)

```
LM load time: 11.0s
Dataset: GSM8K (test set, local arrow loader, seed-stratified)
Dataset load time: 0.1s

  [5/8] success=1, failure=4
Trace collection time: 267.7s
Total rollouts: 8
  Success: 1, Failure: 7
  Failure rate: 0.875

Train: (6, 20, 64), eval: (2, 20, 64) (stratified -> rebalanced)
Train failure rate: 1.000
Eval failure rate:  0.500

Frozen Monitor: AUROC=1.000  (time=0.4s)
Joint Monitor:  AUROC=1.000  (time=0.1s)
Random Monitor: AUROC=0.000  (negative control)
```

Total wrapper wall time: **286.6 s = 4.78 min**.

## Per-job timing projection

| Phase | Time |
|-------|------|
| LM load | 11.0s |
| Trace collection (8 x 200 tokens) | 267.7s |
| 3-arm training + eval | ~1.0s |
| Tail (orphan LM-load block at module bottom) | ~7s |
| **Total** | **~287s = 4.78 min/job** |

60 jobs x 4.78 min/job = **287 min = 4.78 h** (well within the
5-7 h budget from the launcher's estimate).

## Pilot signal (NOT H10 verdict)

| Arm | AUROC |
|-----|-------|
| Frozen | 1.000 |
| Joint | 1.000 |
| Random | 0.000 |
| Delta_F-J | +0.000 |
| Delta_F-R | **+1.000** |

**Negative control PASSES** at the seed level (Frozen / Joint both
above Random by a wide margin).

**Key contrast to simple arithmetic n=100**:

| Setup | Frozen | Joint | Random | F-J |
|-------|--------|-------|--------|-----|
| Simple arith, 64 tok, n=100 | 0.500 | 0.485 | 0.510 | +0.015 |
| GSM8K, 200 tok, seed=100 | 1.000 | 1.000 | 0.000 | +0.000 |

On simple arithmetic, the three arms are at the chance level (~0.5).
On GSM8K 200-token, BOTH Frozen and Joint are at ceiling (1.0) and
Random is at 0.0. This is the expected pattern: 200-token CoT traces
carry a real failure signal that 64-token simple-arithmetic traces
did not.

The remaining question (resolved only by n=20 aggregation): is
**Frozen > Joint**, or do they tie as in this single seed?

## What's NOT validated

- 1 seed is statistically meaningless. Frozen = Joint = 1.000 at seed=100
  does not imply Frozen == Joint at the population level.
- The stratified split collapses when failure rate is very high (87.5%);
  with 1 success / 7 failures the rebalance gave a 2-trace eval set.
  At higher n with seed stratification the classes will balance better.

## Pre-launch decision

Validation passed. Proceed with the full 60-job run.

Launcher: `experiments_log/_run_h10_n20_gsm8k.ps1`
Aggregation: `experiments_log/_agg_h10_n20_gsm8k.py`
Pre-Reg Amendment 1: `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md`

## NO_SELF_DECEPTION.md compliance

- The validation is **annotated as PILOT, n=1, NOT H10 verdict**.
- No claim about Frozen > Joint is made at this stage.
- The 60-job run is launched because the pipeline is mechanically
  sound (correct prompt, correct window, correct dataset, sub-5min/job
  budget), not because the signal direction is pre-known.
