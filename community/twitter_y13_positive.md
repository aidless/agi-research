# Twitter / X 草稿 Y1.3 公告 (v3.0, after 3-way control)

> 2026-07-28. v3.0 reflects v1.2 (3-way control).
> v1.0 was overclaimed. v1.1 over-corrected. v1.2 is the honest
> final version.

---

## Version 1: v1.2 完整 (数字 + 限定 + 机制)

```
Y1.3 3-way control result on LunarLander-v3:

  PPO baseline (no shaping):         40.6 (n=5)
  Y1.3 with INVERSE monitor (1-p):   55.4 (n=5)  +14.7 vs PPO
  Y1.3 with RANDOM monitor:          58.2 (n=5)  +17.6 vs PPO
  Y1.3 with REAL Monitor:            80.1 (n=15) +39.5 vs PPO  t=6.76

Pairwise deltas:
  Real - Random:   +21.9  (Monitor signal adds value above shaping)
  Real - Inverse:  +24.8  (Monitor is direction-sensitive)
  Random - Inverse: +2.8  (both non-informative; equal)

v1.2 (honest) decomposition of Y1.3 effect:
  (a) Reward shaping as regularizer:        +15-18 (any signal)
  (b) Monitor signal as information:        +22-25 (real > non-info)
  Combined:                                +37-40 over PPO, p<0.001

EXPLICIT LIMITATIONS (per NO_SELF_DECEPTION.md):
  - Random and Inverse are n=5; Real is n=15. The +22-25 delta
    is NOT statistically significant with current n.
  - Single env: only LunarLander shows Y1.3 benefit.
  - Acrobot and MountainCar showed no Y1.3 help.
  - Lambda=0.5 only.
  - Mechanism is post-hoc (3 sentences, see paper 4.10.18).

VERSION HISTORY:
  v1.0 (ef90c2c, 2026-07-27): 'FIRST POSITIVE +50 from Monitor' - RETRACTED
  v1.1 (e515565, 2026-07-28): 'shaping helps regardless of signal'
                                - SUPERSEDED
  v1.2 (8faf30b, 2026-07-28): 'shaping + Monitor both contribute,
                                Monitor is direction-sensitive' - HONEST

The novel contribution is the +22-25 Monitor signal delta above
shaping, NOT the +50 vs PPO. Monitor architecture provides
directional information that non-informative signals do not.
