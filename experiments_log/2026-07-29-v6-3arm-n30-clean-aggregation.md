# v6 n=30 3-arm CLEAN: bit-for-bit identity confirmed at n=30 (removes env confound)

> Date: 2026-07-29
> Setup: PettingZoo Simple Spread v3 (continuous, 800 ep/seed)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v6.py`
> 3-arm 30-seed sweep, all 3 arms run with the SAME python (pettingzoo 1.24.3, explicit path)
> Removes the env-inconsistency confound from the n=30 r2/r3 batch.

## Background

The n=30 r3 batch (commit `d7dd99d`) showed 0/30 bit-for-bit identity
between with_verifier and with_trusthead_random, contradicting the n=5
r2 result of 5/5 bit-for-bit identity. The most likely explanation was
that the n=30 r2/r3 batch had a python/pettingzoo environment
inconsistency (the user updated pettingzoo mid-run from 1.24.3 to 1.26.1,
which broke the .mpe submodule).

This r4 batch re-runs the n=30 with_verifier (all 30) + n=30
no_verifier s0-s19 (20) with the EXPLICIT python path
(pettingzoo 1.24.3, which has the .mpe submodule). The
no_verifier s20-s29 and with_trusthead_random s0-s29 from the r3
batch are reused (they were already run with the correct python).

This gives a CLEAN 3-arm 30-seed comparison with consistent
python/pettingzoo/pytorch environment across all 3 arms.

## Results (n=30 per arm, consistent python)

| arm | mean | sd | per-seed source |
|---|---|---|---|
| with_verifier (= v5) | **-69.1715** | 1.9229 | r4 (this batch) |
| no_verifier (v2 baseline) | -69.1299 | 1.9066 | r3 (s20-s29) + r4 (s0-s19) |
| with_trusthead_random | **-69.1715** | 1.9229 | r3 (this arm) |

**with_verifier == with_trusthead_random BIT-FOR-BIT IDENTICAL: 30/30
seeds (100%)**. Max abs diff = 0.000000. Mean abs diff = 0.000000.

Per-seed sanity check (first 5):
| seed | with_verifier | with_trusthead_random | diff |
|---|---|---|---|
| 0 | -68.9809 | -68.9809 | +0.000000 |
| 1 | -66.0427 | -66.0427 | +0.000000 |
| 2 | -70.6001 | -70.6001 | +0.000000 |
| 3 | -66.0383 | -66.0383 | +0.000000 |
| 4 | -67.7251 | -67.7251 | +0.000000 |

## Paired tests (n=30, consistent python)

| comparison | mean_diff | sd_diffs | t | n_pos | sig? |
|---|---|---|---|---|---|
| with_verifier vs no_verifier | -0.0416 | 0.2169 | -1.051 | 14/30 | NOT sig |
| with_trusthead_random vs no_verifier | -0.0416 | 0.2169 | -1.051 | 14/30 | NOT sig |
| with_verifier vs with_trusthead_random | +0.0000 | 0.0000 | nan | 30/30 (eq) | IDENTICAL |

## Bit-for-bit identity across sample sizes

| sample | identical seeds (with_verifier == with_trusthead_random) |
|---|---|
| n=5 (r2) | 5/5 (100%) |
| n=30 (r3, env-inconsistent) | 0/30 (0%) -- contaminated |
| **n=30 (r4, env-consistent)** | **30/30 (100%)** |

**The n=30 r3 result was contaminated by a python/pettingzoo
environment change mid-run.** With consistent environment, the bit-
for-bit identity holds at n=30, just as it did at n=5. The trust
head ignores its input slot, and with_verifier == with_trusthead_random
exactly when the random state is the same.

## What this means

**The trust head architecture does NOT use its input signal.** The
Monitor broadcast, the random uniform, and (in v8) the DLR cross-
agent predicates are all ignored by the trust head. The trust head
learns a function of my_obs only and treats the input slot as
noise.

**The trust head architecture's effect is small and inconsistent.**
At n=5 r2: +0.17 mean over baseline. At n=30 r4: -0.04 mean over
baseline (i.e., trust head actually hurts by 0.04, NOT sig). The
architecture effect shrinks with n, consistent with the v5 n=212
finding (+0.055, 50.5% pos).

**The signal-specific finding (DLR in critic) is now the cleanest
result in the Y2 investigation:**
- v8 dlr_only: +0.1447 (p<0.005, t=+3.216, 20/30 pos at n=30)
- Effect is stable across sample sizes (n=5 +0.15, n=30 +0.14)
- Not contaminated by env issues

## Updated 6-pathway story

| # | path | n | effect | sig? | env? |
|---|---|---|---|---|---|
| 1 | v3 (Monitor aux loss in critic) | 5 | -3.03 | NOT sig, HURTS at 10K | OK |
| 2 | v4 (inter-agent comms in critic) | 5 | +0.00 | NOT sig, no effect | OK |
| 3 | v5 (trust head + same-agent Monitor) | 5/212 | +0.17/+0.055 | NOT sig, shrinks | OK |
| 4 | v6 (trust head + random) | 5/30 | +0.17/0.00 | NOT sig, n=30 bit-for-bit = v5 | OK (r4) |
| 5 | v7 (prior impl, trust head + Monitor) | 5 | 0.00 | Monitor IGNORED | OK |
| 6 | v8 (DLR + trust head) | 30 | +0.00 (= dlr_only) | trust head adds nothing | OK |
| 6' | **v8 dlr_only (DLR in critic only)** | **30** | **+0.1447** | **p<0.005, SIG** | OK |

## Action items

- [x] Rewrite v6 as proper v5 ablation
- [x] n=5 3-arm: confirmed bit-for-bit identity (5/5)
- [x] n=30 3-arm r3: bit-for-bit identity BROKEN (0/30) -- env confound
- [x] **n=30 3-arm r4 CLEAN: bit-for-bit identity RESTORED (30/30)**
- [x] Update 6-pathway paper with r4 finding
- [x] Update H5 in 9-hypo framework
- [x] Commit and push
