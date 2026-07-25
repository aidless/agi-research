# Phase 1 Step 1 smoke (2026-07-25)

> DEC-0008 / DEC-0009 first execution.

## Setup

- Scale: 4 games (coinrun, bigfish, jumper, dodgeball) * 1 seed * 50K env steps
- CPU only (no GPU) via Python 3.10 (Trae Solo CN), procgen 0.10.7
- elapsed: 311s (5.2 min)

## Per-game results

| game    | group | mean ret | median | std | n_ep | p30 threshold |
|---------|-------|----------|--------|-----|------|---------------|
| coinrun | T1    | 5.7      | 10     | ?   | 272  | 0.0            |
| bigfish | T1    | 0.8      | 0      | ?   | 410  | 0.0            |
| jumper  | T1    | 3.3      | 0      | ?   | 115  | 0.0            |
| dodgeball | T1  | 1.1      | 0      | ?   | 423  | 0.0            |

## Interpretation

- All thresholds are 0.0 because **PPO has not yet learned**. Early-game
  policies score mostly 0 reward (or 10 for coinrun's reach-end).
- This is the expected baseline after only ~24 PPO updates per game.
- For Phase 2 (Monitor training) we need policies good enough to produce
  fail / success variance. Plan: increase to ~250K steps for next round.

## Next step

- Decision DEC-0011: Phase 1 Step 2 (= 256K * 3 seeds * 4 games)?
- Or: ship Phase 1 as a calibration dataset and go directly to Phase 2?

## Output

Saved: code/results/procgen/phase1_20260725_100247.json (24 KB)
