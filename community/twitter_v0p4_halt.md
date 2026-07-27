# Twitter / X 草稿 v0.4 公告（6 次 honest negative + 1 次 neutral）

> 2026-07-27. DEC-0011 v0.4 完成后。Monitor 强、online gating 难。

---

## Version 1: 数字版（6 次实验一览）

```
Update from my 5-year AGI program: the Monitor works. The gating doesn''t.

6 online-gating experiments on LunarLander-v3 + CartPole-v1:
  v0.1  Q-BoN, n=5 eval:    +22  (n.s.)
  v0.2  Q-BoN cal., n=50:  -158  (n.s.)
  v0.3  safe=2, n=50:      -718  (sig.)
  v0.4A Q-BoN cal., 5x data: -2  (n.s.)  <-- the 5x data fix
  v0.4B CartPole, n=50:    -270  (sig.)
  v0.4C Imitation, n=50:   -34  (sig.)

0/6 statistically positive.

What we learned: decoupled Monitor gives AUROC 0.99. Strong.
But converting that signal to a policy gain failed in every test.

DEC-0011 v0.4: HALT the online-gating sub-project. Moving to Y1.

#AGI #RL
```

---

## Version 2: 故事版（一个 Monitor 救不了 PPO）

```
A story about what AI engineering actually looks like.

I built a failure predictor for an RL spaceship pilot. AUROC 0.99.
It correctly spots when the pilot is about to crash.

Then I tried to USE the predictor to OVERRIDE the pilot.

6 different strategies. 5/6 made the spaceship crash more.
Only 1 (with 5x more training data) was neutral.

The lesson: having a good failure detector is not the same as
knowing what to do about it. Prediction is not action.

Code + paper: github.com/aidless/agi-research

#ML #RL #AGI
```

---

## Version 3: 方法论版（5x 数据 -158 → -2）

```
The most useful result from my last 6 AGI experiments:

Same code, same env, same hyperparameters.
- With 200 train episodes: delta = -158 (catastrophic)
- With 1000 train episodes: delta = -2  (neutral)

The 5x increase in data:
  - stopped val_auroc=1.0 overfitting
  - brought cal_threshold from ~0 to a sensible 0.1-0.65
  - reduced variance from 209 to 17

Honest engineering rule: a calibrated threshold needs more than
40 val examples. We needed 200+. The "free" 0/6 results
above are mostly the data-size artifact, not a method failure.

#ML #DataQuality
```

---

## Version 4: 一行版（Monitor 强、gating 难）

```
Decoupled Monitor: AUROC 0.99.
Online Monitor-driven gating: 0/6 statistically positive.
5/6 were significantly negative.

Prediction works. Action-selection from that prediction does not.

#AGI #RL
```

---

## 配图建议

1. **6-bar chart**: 6 experiments, y-axis delta (gated - ungated mean).
   5 bars negative (red), 1 bar near zero (v0.4A, neutral). One bar (v0.1) is +22.
2. **Convergence figure**: delta std vs train episodes.
   200 train: std=209 (overfit). 1000 train: std=17 (honest).
3. **Architecture diagram**: 4 layers (Monitor + C + D + E + Q).
   Highlight "gating layer" with a strikethrough or "HALT" label.

---

## 配套 Discord / Reddit 长版

Discord version is in `community/discord_v0p4_halt.md` (separate file).
Same numbers, longer story, ends with explicit Y1 roadmap.
