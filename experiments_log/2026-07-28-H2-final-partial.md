# H2 PRELIMINARY verdict - Acrobot n=5 (real vs PPO)

> Date: 2026-07-28
> Status: H2 pre-registered, full sweep aborted (processes killed).
>         PRELIMINARY verdict using existing cross-env test data.
>         Final H2 verdict (real vs random, n=10) PENDING.

## 1. Preliminary H2 data (real vs PPO, n=5 each)

| Method | n | Mean | Std | Per-seed |
|--------|---|------|-----|----------|
| Y1.3 (real) | 5 | -91.7 | 13.3 | [-115, -87, -88, -87, -81] |
| PPO baseline | 5 | -87.4 | 8.4 | [-80, -87, -85, -102, -83] |

**Delta: -4.3** (Y1.3 slightly worse)
**Welch t: -0.61** (NOT significant)

## 2. Cross-env H2 trend

| Env | Y1.3 vs PPO | Real - Random | Verdict |
|-----|-------------|---------------|---------|
| LunarLander-v3 | +39.5 (sig, t=1.94) | +13.6 (NOT sig) | Monitor not validated |
| Acrobot-v1 | -4.3 (NOT sig) | (PENDING, sweep aborted) | Y1.3 doesn't help |

**Pattern**: Y1.3 with real Monitor does NOT significantly help
PPO on either env. The +50 from v1.0 was specifically against
LunarLander PPO baseline; against random shaping on the same env
the effect is +13.6 (NOT sig). On a different env (Acrobot) the
real effect is essentially zero.

## 3. What H2 (preliminary) tells us

- The Monitor signal does NOT help PPO on Acrobot (delta=-4.3)
- This is consistent with H1 verdict (Monitor not validated on
  LunarLander above shaping)
- Two-env evidence supports the v1.3 conclusion: Y1.3's
  contribution is "reward shaping helps" not "Monitor helps"
- The Monitor architecture is real (AUROC 0.99) but does not
  transfer to policy improvement at this PPO budget

## 4. Decision record (DEC-Y1.3 v1.4 PRELIMINARY)

H1 (LunarLander, n=15 per arm): H1 NOT supported.
H2 PRELIMINARY (Acrobot, real vs PPO n=5): Y1.3 does not help.
  H2 verdict on real vs random: PENDING (sweep aborted).

The Y1.3 sub-project is now well-characterized:
  - Real Monitor signal: NOT validated above random shaping (H1)
  - Real Monitor signal: NOT different from PPO on Acrobot (H2 prelim)
  - Y1.3 contribution: reward shaping regularizer (any signal)

## 5. Next steps

1. **Re-run H2 sweep with fewer parallel processes** (5 instead
   of 10) to avoid OOM. The full H2 verdict (real vs random on
   Acrobot, n=10 each) is still pending.
2. **Or accept the preliminary verdict** as the H2 result. The
   trend across 2 envs is consistent: Y1.3 does not provide
   publishable policy improvement above PPO baseline.
3. **Move to next direction**: H3 (different PPO budget, model-based
   planning, expert imitation, etc.)

I recommend (3): the Y1.3 sub-project has 2 envs of negative evidence.
Further extending the same intervention is unlikely to flip the
verdict. New directions (different intervention, different env,
different PPO variant) are more likely to be productive.
