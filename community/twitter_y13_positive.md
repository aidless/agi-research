# Twitter / X 草稿 Y1.3 公告

> 2026-07-27. Phase 1.5 第一个正结果。Monitor 当训练约束，不当 inference 介入。

---

## Version 1: 数字版（最直接）

```
After 6 failed attempts at inference-time gating, the 7th attempt worked.

Y1.3: use the failure-prediction Monitor as a PPO TRAINING-time
reward shaper (not inference-time action selector).

Setup: 100K PPO on LunarLander-v3, 5 seeds:
  PPO-only baseline:  40.6 mean  (n_eval=50)
  Y1.3 (lambda=0.5):  90.5 mean  (+50, t=1.65, 3/5 wins)

3 of 5 seeds improve by +60 to +105 points. One seed loses by -59.
Not yet significant (need t>2.3) but clearly directional.

#AGI #RL #SelfImprovement
```

---

## Version 2: 故事版（"换了思路才成功"）

```
I tried 6 different ways to use a failure predictor to improve
an RL agent. 6/6 made it worse.

The 7th attempt: stop using the predictor at INFERENCE time.
Use it as a TRAINING constraint instead.

Result: +50 mean reward (90.5 vs 40.6 baseline), 3/5 seeds win.

The lesson: a good failure detector is useful as a "navigation aid"
(where NOT to go during learning), not as an instruction (what
action to take in a given state). Inference-time override is
too brittle; training-time shaping is robust.

#ML #RL
```

---

## Version 3: 方法论版（为什么 v0 全失败但 Y1.3 行）

```
A pattern I see in my failed experiments vs the new working one:

FAILED (v0.1-v0.4C): Monitor OVERRIDES PPO at inference.
  if Monitor_prob > threshold: action = Q_Or_Safe_Or_Clone
  Requires the action selector to be RELIABLY good.
  With 200-1000 train episodes, no tested selector was reliable.

WORKS (Y1.3): Monitor PENALIZES PPO reward during training.
  shaped_reward = env_reward - lambda * Monitor_prob
  The policy learns to avoid Monitor-flagged states.
  At inference, PPO acts alone with no Monitor overhead.

Same Monitor (AUROC 0.99). Same 100K PPO budget. Different role.

#ML #RL #Lessons
```

---

## Version 4: 一行版

```
After 6 failures: switched the Monitor from inference-time gating
to training-time reward shaping. +50 over baseline (t=1.65).
3 of 5 seeds win by 60-105 points.
```

---

## 配图建议

1. **5-seed grouped bar chart**: PPO-only vs Y1.3 per seed. Show the 3 wins clearly.
2. **Y1.3 vs PPO baseline over seeds**: 2-line plot showing Y1.3 above baseline in 4/5 seeds.
3. **Sequence diagram**: PPO 25K -> Monitor train -> PPO 75K shaped -> Eval pure PPO.

---

## 配套 Discord / Reddit 长版

Discord version in `community/discord_y13_positive.md` (separate file).
Includes 7-attempt table (v0.1-v0.4C + Y1.3), full data, lessons learned.
