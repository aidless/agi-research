# Phase 1.5 Y1.3 REVISED — Random and Inverse controls

> Date: 2026-07-28
> Status: v1.0 was overclaimed. v1.1 incorporates negative + inverse
>         controls. The original "FIRST POSITIVE result" framing is
>         revised to a more honest "reward shaping helps" framing.

## 1. The original v1.0 claim (RETRACTED in v1.1)

Commit `ef90c2c` and `416d0d7` claimed:
  "Y1.3 (Monitor as PPO training-time regularizer) is the FIRST
   positive result. +50 over PPO baseline, t=6.76 (p<0.001)."

This was self-deceptive per the knowledge base critique pattern:
  - Single baseline (PPO-only, not random-shaping)
  - No negative control
  - No mechanism explanation
  - No multi-env replication
  - t-statistic against a convenient baseline

## 2. The v1.1 controls

To address these, we ran TWO control experiments:

### 2.1 Random monitor control (5 seeds)
- Same Y1.3 pipeline but with `monitor_prob = uniform[0, 1]` (random)
  instead of the trained Monitor
- Result: **58.2 +/- 51.7** (n=5)
- This is +17.6 over PPO baseline (40.6) and **only -21.9 below** Y1.3
  real Monitor (80.1, n=15)
- Interpretation: most of the +50 effect is from **reward shaping**,
  not from the **Monitor signal being informative**

### 2.2 Inverse monitor control (5 seeds, RUNNING)
- Same Y1.3 pipeline but with `monitor_prob = 1 - real_monitor_prob`
  (inverse direction)
- If this gives similar +50: the Monitor signal is uninformative,
  only the shaping matters
- If this gives 0 or negative: the Monitor is direction-sensitive
- If this gives > +60: real Monitor was meaningfully better than inverse

## 3. v1.1 verdict (3-way comparison)

| Method | n | Mean | Std | Delta vs PPO |
|--------|---|------|-----|--------------|
| PPO baseline (no shaping) | 5 | 40.6 | 37.1 | - |
| Y1.3 with **RANDOM** monitor | 5 | 58.2 | 51.7 | +17.6 |
| Y1.3 with **INVERSE** monitor | 5 | (pending) | - | (pending) |
| Y1.3 with **REAL** Monitor | 15 | 80.1 | 45.9 | +39.5 |

## 4. REVISED interpretation

The +50 effect (real Monitor vs PPO baseline) decomposes:
  - ~+18 from the shaping procedure itself (any signal source)
  - ~+22 from the Monitor signal being directionally informative
    (real vs random: +22, but n=5 vs n=15, not statistically significant)

The **correct, honest** claim is:
  "Reward shaping during PPO training (regardless of signal source)
   helps LunarLander PPO by roughly +18 to +40 mean reward. The
   specific Monitor signal adds only marginal extra value (+22) that
   is not distinguishable from chance at the current sample size."

The **incorrect, overclaimed** v1.0 claim was:
  "Monitor as PPO training-time regularizer gives +50 over PPO,
   t=6.76, p<0.001." (implying Monitor is the cause)

## 5. Action taken

1. **NO_SELF_DECEPTION.md** created as a hard protocol for future work
2. **DEC-Y1.3 v1.1** written to document the correction
3. **y13_negative_control.py** added (random signal)
4. **y13_inverse_control.py** added (inverse signal, running)
5. **Twitter/Discord drafts REVISED** to include limitations
6. **Paper Section 4.10 will be updated** with the v1.1 framing
7. **All future claims** must pass NO_SELF_DECEPTION.md P0 checklist

## 6. What I would have done differently

The original ef90c2c commit should NOT have been pushed without:
  - Running y13_negative_control.py first
  - Comparing against y13_random (not just PPO baseline)
  - Writing a mechanism hypothesis before the data
  - Stating limitations in the Twitter/Discord drafts

This is the lesson. Future work follows NO_SELF_DECEPTION.md
mandatory P0 checklist.
