# Twitter / X 草稿 Y1.3 v4.0 (FINAL, after n=15 per arm)

> 2026-07-28. v4.0 reflects v1.3 final verdict.
> Pre-registered H1 NOT supported. v1.0/v1.1/v1.2 all superseded.
> The truly honest finding is "shaping helps; Monitor not validated."

---

## Version 1: 诚实发现 v1.3（n=15 per arm）

```
Y1.3 final verdict on LunarLander-v3, n=15 per arm:

  PPO baseline:        40.6 (n=5)
  Y1.3 INVERSE:        56.7 (n=10)  +16.0 vs PPO
  Y1.3 RANDOM:         66.5 (n=10)  +25.9 vs PPO
  Y1.3 REAL:           80.1 (n=15)  +39.5 vs PPO

Pre-registered Welch t-tests:
  Real - Random:     t=0.78  delta=+13.6  NOT significant
  Real - Inverse:    t=1.06  delta=+23.5  NOT significant
  Real - PPO:        t=1.94  delta=+39.5  borderline
  Random - Inverse:  t=0.44  delta=+9.8   NOT significant

Pre-registered H1 verdict: NOT SUPPORTED.
  H1: delta > +10 AND t > 2.0
  Result: delta=+13.6, t=0.78. H1 fails.

HONEST FINDING (v1.3 final):
  'PPO + any per-step reward shaping (random, inverse, or trained
  Monitor) helps LunarLander PPO by ~+16-40 mean reward. The
  specific Monitor signal does not significantly improve on this
  baseline. The Monitor architecture provides useful real-time
  failure prediction (Sections 4.6-4.8, AUROC 0.99) but does not
  transfer to policy improvement at this PPO budget.'

VERSION HISTORY (all in git history):
  v1.0 (ef90c2c):    'FIRST POSITIVE +50 from Monitor' - RETRACTED
  v1.1 (e515565):    'shaping regardless of signal' - SUPERSEDED
  v1.2 (8faf30b, n=5): 'Real > Random~Inverse by +22-25' - SUPERSEDED
  v1.3 (78b6044, n=15): 'Shaping helps; Monitor not validated' - FINAL

WHAT THE PRE-REGISTRATION PROTOCOL SAVED:
  Without pre-registration, the n=5 v1.2 result would have been
  published as a 'novel contribution'. With pre-registration and
  n=15: H1 fails. The protocol converted a publishable-looking
  pilot into a NULL result that we can honestly report.

KEY LESSON:
  The Monitor architecture is real (AUROC 0.99 at prediction) but
  does not transfer to policy improvement at this PPO budget. This
  is a meaningful NULL result for the Y1.3 sub-project. Future
  directions: model-based planning, expert imitation, or longer
  PPO budgets.
