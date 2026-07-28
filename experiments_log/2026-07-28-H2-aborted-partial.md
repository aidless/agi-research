# H2 Acrobot sweep ABORTED - process killed during execution

> Date: 2026-07-28
> Status: H2 sweep launched but killed before completion. Partial
>         data available; full H2 verdict pending.

## 1. What happened

H2 sweep launched at 11:20:31 on 2026-07-28:
  - 5 Acrobot Y1.3 (real, out-tag y13acrobat) - new run
  - 5 Acrobot Y1.3 (random, out-tag y13ncacrobot) - new run
  - Pre-registered H2: Acrobot n=10 per arm, real vs random

20 parallel processes (10 launcher parents + 10 python children).
Processes ran for ~10-15 min (CPU ~200-300 sec each), then died
without producing JSON. All `.err` files are 0 bytes (no Python
exception). All python processes have Count=0 now.

Most likely cause: system OOM kill (20 parallel python processes
each ~500 MB = ~10 GB, plus 4 GB for other system = borderline).

## 2. Partial data available

The 5 y13acrobot (real) JSONs that the new H2 sweep was supposed
to write do NOT exist (the new sweep didn't complete). However,
the original cross-env test (commit 416d0d7, 2026-07-27) already
ran 5 y13acrobot (real) and 5 ppoacrobat (PPO baseline) on Acrobot
in the earlier cross-env test. Those are saved as:
  - checkpoints/full_integration_y13acrobat_Acrobot-v1_seed{0..4}/phase2_log.json
  - checkpoints/full_integration_ppoacrobat_Acrobot-v1_seed{0..4}/phase2_log.json

Those give us a PRELIMINARY H2 verdict (real vs PPO, no random).

## 3. Preliminary H2 verdict (real vs PPO, n=5 each, Acrobot)

To be aggregated in next step. The pre-registered H2 verdict
(real vs random, n=10 each) is PENDING.

## 4. Next steps

1. Re-run H2 sweep with fewer parallel processes (e.g., 5 at a time
   instead of 10) to avoid OOM
2. OR run H2 sweep sequentially (~50-100 min total)
3. OR use existing 5 y13acrobot (real) + 5 ppoacrobat (PPO) +
   add 5 more random control to get a partial H2 (real vs PPO + 5
   real vs 5 random)

I will proceed with (3) since it's cheapest. If H2 is unclear
with this data, we can do (1) or (2) next.
