# Twitter / X 草稿 Y1.3 公告 (REVISED, with explicit limitations)

> 2026-07-28. v2.0 (REVISED with limitations per NO_SELF_DECEPTION.md).
> Original v1.0 (2026-07-27) was overclaimed. v2.0 adds explicit
> limitations and conditions for "POSITIVE result" status.

---

## Version 1: 数字 + 限定条件（v2.0 REVISED）

```
Y1.3 result on LunarLander-v3 (n=15 seeds, 100K PPO each):

  Y1.3 (Monitor as PPO training-time regularizer):
    Mean: 80.1 +/- 45.9,  t=6.76 (p<0.001)

  PPO-only baseline (n=5):
    Mean: 40.6 +/- 37.1

  Delta: +39.5

EXPLICIT LIMITATIONS (per NO_SELF_DECEPTION.md):

  (1) Single environment: only LunarLander shows significance.
      Acrobot is neutral (-88.7 vs -87.4), MountainCar fails at
      100K PPO regardless of Y1.3.

  (2) No negative control: I have NOT yet run Y1.3 with a RANDOM
      monitor signal in place of the trained Monitor. If random
      monitor gives the same +50 mean, the result is not due
      to the Monitor signal. Negative control is RUNNING as of
      2026-07-28; results pending.

  (3) No mechanism explanation: I do not know WHY Y1.3 works.
      Possible mechanisms: (a) Monitor pushes policy away from
      failure-like states, (b) monitor noise provides regularization,
      (c) PPO benefits from any reward perturbation. Mechanism
      work pending.

  (4) Not pre-registered: H1 was not stated before data was
      collected. This is a known limitation of the v1.0 result.

  (5) DEC-Y1.3 v1.1 sets explicit publishability criteria:
      result is "publishable" only when (1)-(4) are all addressed.
      See experiments_log/2026-07-28-DEC-Y1.3-v1.1-publishability.md

What I will do next:
  - Run the negative control (5 seeds, ~30 min)
  - If control = Y1.3: reframe the claim, do NOT call it "due to Monitor"
  - If control < Y1.3: original claim supported, write mechanism,
    run 1 more env, REVISED announcement
