# Pre-registered H2.0-B - Simple MLP Monitor (ablation of slot attention)

> Date: 2026-07-28
> Purpose: per NO_SELF_DECEPTION.md, ablate the slot-attention
>         architecture. Does a SIMPLER Monitor (MLP) work better
>         than the COMPLEX slot-attention Monitor? This is an
>         architecture ablation test, not a new intervention.

## 1. Background

The Y1.x Monitor used slot attention + a small MLP head (see
y13_monitor_regularizer.py):
  - slot = SlotAttention(n_slots=4, slot_dim=32, n_iters=3, hidden=64, input_dim=per_step)
  - monitor = MLP head on top of slot output
  - 4 H tests showed this Monitor does NOT help PPO

H2.0-B tests whether a SIMPLER Monitor (just a 2-layer MLP on the
flattened trajectory) would work BETTER. If yes, this is a
"simpler is better" finding. If no, it confirms the failed Y1.x
verdict generalizes across architectures.

## 2. Hypothesis (PRE-REGISTERED)

**Env**: LunarLander-v3 (same as H1, H1.4, H2.0-A)

**H2.0-B**: With 100K PPO budget and lambda=0.5, Y1.3 with a
  SIMPLE MLP Monitor (no slot attention) gives a higher mean
  return than Y1.3 with a random monitor of the same signal
  magnitude, with delta > +10 AND Welch t > 2.0.

**H0 (null)**: Simple MLP Monitor and random monitor give same mean
  return (delta < +10 or t < 2.0).

**Decision rule**: Same as H1/H2/H3/H1.4.
  - If H2.0-B supported: "Simple MLP Monitor is informative above
    random" (suggests slot attention was the problem)
  - If H0 supported: "Monitor architecture doesn't matter; even
    simple MLP doesn't help PPO"

**Why H2.0-B is worth testing separately from Y1.x**:
- Tests if simpler architecture is better (Occam's razor)
- Different intervention class: ablation, not new use case
- If H2.0-B supports: the slot attention is the bottleneck, not
  the Monitor signal
- If H2.0-B NOT supported: even simple MLP doesn't help, so the
  failure mode is not architectural complexity

## 3. Pre-registered sample size

n=5 per arm (simple MLP real vs simple MLP random). Matches H3/H1.4.

## 4. Pre-registered exclusion rules

A seed is excluded ONLY if:
  - PPO training crashes
  - Eval episodes truncated
  - Seed number set wrong

## 5. Pre-registered analysis plan

For each arm:
  - Per-seed mean eval return (50 episodes)
  - Aggregate mean, std
For comparison:
  - Welch t-test
For the verdict:
  - If Real MLP > Random with t > 2.0 and delta > +10: claim
    "Simple MLP Monitor is informative above random"
  - Otherwise: claim "H2.0-B NOT supported; simple MLP doesn't
    help either"

## 6. Pre-registered stopping rule

Run to completion (n=5 per arm) without interim peeking.

## 7. Comparison to Y1.x

If H2.0-B supports (simple MLP helps, slot attention didn't), this
would suggest the slot attention architecture was the bottleneck.
If H2.0-B doesn't support, this confirms the Y1.x verdict that
"the Monitor signal does not help PPO" regardless of architecture.

## 8. Pre-registration log

H2.0-B was registered on 2026-07-28 BEFORE the sweep was launched.
Any change to the registered hypothesis, sample size, decision
rule, or stopping rule must be documented as a deviation and
justified.
