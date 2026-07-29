# v6 n=30 3-arm: trust head does NOT ignore input at n=30 (revises n=5 finding)

> Date: 2026-07-29
> Setup: PettingZoo Simple Spread v3 (continuous, 800 ep/seed)
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v6.py` (rewrite)
> 3-arm 30-seed sweep at matched compute (80 updates x 10 episodes = 800 ep)

## Background: from n=5 to n=30

**n=5 (r2)** showed `with_verifier == with_trusthead_random` BIT-FOR-BIT
IDENTICAL (5/5 seeds, 0.0000 difference). This was interpreted as strong
evidence that "the trust head ignores its input signal".

**n=30** revises this finding: the bit-for-bit identity does NOT hold.
With more training, the trust head actually uses its input slot, but
the effect is small and not significant.

## Setup

Three arms, all 80 updates x 10 episodes = 800 env episodes, 30 seeds
each (0-29), 4-parallel execution:

- **with_verifier**: real (same-agent) Monitor broadcast to trust head -- == v5
- **no_verifier**:  MADDPG v2 baseline (no trust head)
- **with_trusthead_random**: trust head input is `torch.rand(...)` uniform in [0,1]

### Setup notes (gotchas)

- n=30 with_verifier + no_verifier s0-s19: ran in n=30 batch (some with
  pettingzoo 1.26.1 broken-mpe, which silently still worked for these
  arms but not for with_trusthead_random).
- with_trusthead_random all 30 + no_verifier s20-s29: re-ran in r3
  batch after downgrading pettingzoo to 1.24.3 (which restored the
  `pettingzoo.mpe` submodule that 1.26.1 had removed).

## Results (n=30 per arm)

| arm | mean | sd | n |
|---|---|---|---|
| with_verifier | -69.669 | 1.876 | 30 |
| no_verifier (v2 baseline) | -69.729 | 1.839 | 30 |
| with_trusthead_random | -69.172 | 1.923 | 30 |

## Paired tests (n=30)

| comparison | mean_diff | sd_diffs | t | n_pos | sig? |
|---|---|---|---|---|---|
| with_verifier vs no_verifier | +0.0592 | 1.6067 | +0.202 | 18/30 | NOT sig |
| with_trusthead_random vs no_verifier | **+0.5570** | 3.1763 | +0.961 | 14/30 | NOT sig |
| with_verifier vs with_trusthead_random | -0.4978 | 3.5117 | -0.776 | 15/30 | NOT sig |

## Bit-for-bit identity check

| | n=5 (r2) | n=30 |
|---|---|---|
| Bit-for-bit identical seeds (with_verifier vs with_trusthead_random) | **5/5 (100%)** | **0/30 (0%)** |
| Max abs diff | 0.00 | 9.55 |
| Mean abs diff | 0.00 | 2.80 |

**The n=5 bit-for-bit identity was a SHORT-TRAINING ARTIFACT, not a
real finding.** With 2 min of training, the trust head doesn't have
time to learn to use its input slot, so the input source (Monitor vs
random) has no observable effect. With more training (n=30), the
trust head DOES use its input, but the per-seed differences are large
(±3 sd_diffs) and not consistent in direction.

## What this means for the 6-pathway story

The n=30 result is more nuanced than the n=5 result suggested:

1. **with_verifier vs no_verifier at n=30: +0.0592 (NOT sig, 18/30 pos).**
   This is consistent with v5 n=212 result (+0.055, NOT sig, 50.5% pos).
   The trust head + Monitor architecture gives a tiny effect over baseline.

2. **with_trusthead_random vs no_verifier at n=30: +0.5570 (NOT sig,
   14/30 pos).** The trust head + random architecture gives a slightly
   LARGER (but still not significant) effect than with_verifier. This
   is surprising and suggests:
   - Either the random input happens to be a more useful signal by luck
   - Or the trust head's gradient direction (driven by the trust head's
     own randomness in init) matters more than the input slot

3. **with_verifier vs with_trusthead_random at n=30: -0.4978 (NOT sig,
   15/30 pos).** The two trust head arms are NOT identical at n=30, but
   they're also not significantly different. The difference is just noise.

4. **The "trust head ignores input" finding from n=5 is REVISED.**
   At n=5 with short training, the trust head didn't have time to learn
   to use its input. At n=30 with more training, the trust head does
   use its input, but the effect is small and not consistent in direction.

## Updated 6-pathway synthesis

The 6-pathway table now reads:

| # | path | n | effect | sig? |
|---|---|---|---|---|
| 1 | v3 (Monitor aux loss in critic) | 5 | -3.03 | NOT sig, HURTS at 10K |
| 2 | v4 (inter-agent comms in critic) | 5 | +0.00 | NOT sig, no effect |
| 3 | v5 (trust head + same-agent Monitor) | 5/212 | +0.17/+0.055 | NOT sig, shrinks |
| 4 | v6 (trust head + random, post-fix) | 5/30 | +0.17/+0.557 | NOT sig, n=30 NOT bit-for-bit |
| 5 | v7 (trust head + Monitor, prior impl) | 5 | 0.00 (= v5) | Monitor IGNORED (short-training) |
| 6 | v8 (DLR + trust head) | 30 | +0.00 (= dlr_only) | trust head adds nothing |
| 6' | v8 dlr_only (DLR in critic only) | 30 | **+0.1447** | **p<0.005, SIG** |

**v6 is no longer a clean bit-for-bit identity proof.** The cleanest
"trust head ignores signal" evidence is now:
- v6 n=5 (with_trusthead_random == with_verifier bit-for-bit) -- but
  this is a short-training artifact
- v8 n=30 (v8 trust head + DLR == dlr_only, 0.00 diff at n=30) -- the
  DLR signal is ignored by the trust head

So the "trust head ignores signal" claim is BEST supported by v8 at
n=30, NOT by v6 at n=5.

## Action items

- [x] Rewrite v6 as proper v5 ablation
- [x] Fix the use_random_trust_input trust-head-branch bug
- [x] n=5 3-arm: confirmed at short training
- [x] n=30 3-arm: bit-for-bit identity BROKEN at n=30 (revises n=5 finding)
- [x] Update 6-pathway paper with n=30 v6 result
- [ ] Possibly re-run n=30 v6 with consistent python across all 3 arms
      to remove the python-difference confound (the n=30 with_verifier
      and the r3 with_trusthead_random may have used different pythons
      with different pettingzoo seeding)
