# B. Multi-seed LunarLander Phase 2 (today)

## Setup
- env: LunarLander-v3, 256K PPO, threshold capped at 0
- 3 seeds: 0, 1, 2
- 200 train / 100 eval episodes each
- same Monitor architecture (64-64-64 MLP)

## Per-seed results

| seed | PPO mean | p10 (capped) | train fail | train AUROC | eval AUROC | eval mean prob std |
|------|----------|--------------|------------|-------------|-------------|---------------------|
| 0    | +142.8   | 0.0          | 0/200      | (skipped, 0 fails) | **0.980** | 0.003 |
| 1    | +143.8   | 0.0          | 45/200     | (very high)  | **0.899** | 0.02  |
| 2    | (converged) | 0.0        | (some)     | (trained)    | **0.212** | (varies) |

## Aggregate

mean AUROC across 3 seeds = 0.70 (high variance: range 0.21-0.98)

## Interpretation

**H1 directional support is NOT seed-robust.**

- Seed 0: 0.98 (strong, near-perfect)
- Seed 1: 0.90 (strong)
- Seed 2: 0.21 (BELOW chance; Monitor is INVERSE of true failure)

The Monitor is not learning a robust representation. It is learning
PPO-instance-specific patterns that happen to correlate with
success/failure on some seeds but not others.

## What this means for the paper

- H1 directional claim has 2/3 positive seeds. We need to either:
  (a) Add more seeds (5+) for statistical significance
  (b) Investigate why seed 2 produces INVERSE correlation
  (c) Use a different Monitor architecture (e.g., LSTM, transformer)

- The high variance is itself a publishable finding. It means
  decoupling helps on average but the magnitude is unstable.
- For paper, this is an honest "results are seed-dependent" report.

## Open

- D: adversarial perturbation test (next)
- Audit + tech doc (after D)
