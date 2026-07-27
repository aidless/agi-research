# Discord / Reddit 长版 Y1.3 公告

> 2026-07-27
> Audience: AGI/RL researchers, my own future self, critique partners

---

## Headline

**After 6 failed attempts at online gating, attempt 7 (training-time
regularization) works. +50 over PPO baseline, 3/5 seeds win.**

---

## TL;DR

In my Phase 1.5 (4-layer AGI integration) program, I have a decoupled
failure Monitor (SlotMonitor) that achieves AUROC 0.99 on LunarLander-v3.
I tried 6 different ways to use this Monitor for inference-time action
selection (Q-BoN, calibrated Q-BoN, fixed safe action, behavior cloning,
on LunarLander and CartPole). **All 6 failed** to reliably improve the
PPO baseline.

**Y1.3** (attempt 7) uses the Monitor as a TRAINING-time reward shaper
instead of an inference-time action selector. The policy learns to
navigate away from Monitor-flagged states, but at deployment it acts
alone with no Monitor overhead. **Y1.3 > PPO-only baseline by +50
mean (90.5 vs 40.6), 3/5 seeds win by 60-105 points** (t=1.65, p>0.05
but directional).

---

## The 7-attempt journey

| # | Setting | Result | n_train | t | Pos |
|---|---------|--------|---------|---|-----|
| v0.1 | Q-BoN, fixed threshold=0.5 | +22 mixed | 200 | 0.72 | 3/5 |
| v0.2 | Q-BoN, calibrated threshold | -158 | 200 | -1.69 | 0/5 |
| v0.3 | safe action (main engine) | -718 | 200 | -3.71** | 0/5 |
| v0.4A | Q-BoN, calibrated, 5x data | -2 neutral | 1000 | -0.25 | 3/5 |
| v0.4B | Q-BoN, calibrated, CartPole | -270 | 200 | -3.48** | 0/5 |
| v0.4C | Behavior cloning (BC) | -34 | 200 | -2.64** | 0/5 |
| **Y1.3** | **Monitor as reward shaper (training-time)** | **+50** | 200 | **1.65** | **3/5** |

The 6 failures share a common pattern: they all use the Monitor to
OVERRIDE PPO at inference. They differ only in HOW they choose the
override action (Q, safe, BC, calibrated thresholds). The bottleneck
was always the action selector, not the Monitor.

Y1.3 sidesteps this by using the Monitor as a TRAINING signal only.
The policy learns to AVOID high-Monitor states. At deployment, no
Monitor is needed - the policy has internalized the avoidance.

---

## Y1.3 per-seed data

| Seed | Y1.3 mean | Y1.3 std | PPO-only mean | PPO-only std | Delta |
|------|-----------|----------|---------------|--------------|-------|
| 0    | 75.6      | 56.5     | 11.4          | 22.9         | +64.2 |
| 1    | 29.2      | 21.6     | 87.9          | 73.6         | -58.7 |
| 2    | 105.2     | 74.9     | 20.4          | 66.8         | +84.8 |
| 3    | 178.7     | 114.1    | 73.3          | 75.3         | +105.4|
| 4    | 63.8      | 43.1     | 10.2          | 51.7         | +53.6 |

3 of 5 seeds improve substantially (largest +105). One seed loses
moderately (-59). The pattern is consistent: when Y1.3 helps, it
helps a lot; when it does not, the loss is bounded.

---

## Why Y1.3 works

1. **No online decision required.** Inference-time gating failed
   because the action selector was bad. Y1.3 does not need an
   action selector at all.

2. **The Monitor signal is a useful regularizer.** With Monitor
   AUROC 0.99, `Monitor_prob(window)` is a high-quality proxy for
   "how risky is this trajectory?". Subtracting it from the reward
   pushes the policy toward low-risk behaviors.

3. **The constraint is local and smooth.** Each shaped reward is
   a small, smooth function of the recent trajectory. PPO can
   optimize against it without disrupting the rest of the policy.

---

## Why Y1.3 is not yet "publishable"

- t=1.65 is not significant at p<0.05 (need t>2.3 for df~8).
- The effect is variable: some seeds see +100, others see -60.
- The Monitor architecture (slot-attention + small MLP) and the
  reward shaping (subtraction) are both simple. There may be
  better choices (e.g., multiplicative penalty, larger Monitor
  signal, adaptive lambda per state).

I am currently running a lambda sensitivity sweep (lambda=1.0, 2.0,
5.0) to see if stronger shaping makes the effect more consistent.
Results in ~30 min.

---

## What I am NOT saying

- Y1.3 is not a "breakthrough." It is a directionally positive result
  in a 7-attempt sequence where 6 failed. The Monitor is real
  (AUROC 0.99) but converting it to a reliable policy gain is
  still hard.

- The Monitor is not "solved" as a training signal. Y1.3 just
  shows that one specific use (reward subtraction, lambda=0.5,
  slot-attention Monitor) works directionally on LunarLander-v3.
  Other envs, lambdas, and Monitor architectures remain untested.

- The decoupling contribution (H1: Monitor > joint) is unchanged.
  H1 is supported at the PREDICTION level. The POLICY-level
  interventions (v0.1-Y1.3) are a separate sub-project.

---

## Next steps

1. **Lambda sensitivity**: lambda=1.0, 2.0, 5.0 (running now).
   Goal: identify lambda that maximizes mean and minimizes variance.
2. **More seeds**: if lambda=1.0 looks better, run 10-20 seeds.
   Goal: t > 2.3 for statistical significance.
3. **Different env**: Acrobot, MountainCar, simple gridworlds.
   Goal: see if Y1.3 generalizes beyond LunarLander.
4. **Architecture comparison**: try a simple MLP Monitor (no
   slot attention). Goal: see if the slot-attention structure
   matters or if any good Monitor works.

Code + paper: github.com/aidless/agi-research (MIT, attribution 刘泽文)

---

## How to reproduce

```bash
# Treatment
python projects/project_a_self_improvement/code/y13_monitor_regularizer.py \
  --n-ppo-steps-total 100000 --n-warmup-steps 25000 \
  --n-train-episodes 200 --n-eval-episodes 50 \
  --history-len 32 --seed 0 --monitor-lambda 0.5 --out-tag y13

# Control
python projects/project_a_self_improvement/code/ppo_only_baseline.py \
  --n-ppo-steps 100000 --n-eval-episodes 50 --seed 0 --out-tag ppobase
```

Per-seed JSON: `checkpoints/full_integration_y13_LunarLander-v3_seed0/phase2_log.json`
Per-seed raw log: `experiments_log/_y13_seed0.log`

Paper: `projects/project_a_self_improvement/paper_v2_full.md`
  - Section 4.6-4.8: H1 (Monitor prediction) - supported
  - Section 4.10.1-4.11: v0.1-v0.4C failures (6 attempts)
  - Section 4.10.12-4.10.14: Y1.3 (this announcement)

Critique partners welcome: feedback on Y1.3 design or the v0
failure modes is appreciated.
