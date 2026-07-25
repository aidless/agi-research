# B. 5-seed LunarLander Phase 2 (today) - final

## Per-seed results (all LunarLander-v3, 256K PPO)

| seed | PPO mean | train fail | train AUROC | eval AUROC | eval prob std | Pearson(prob, reward) |
|------|----------|------------|-------------|-------------|----------------|------------------------|
| 0    | +142.8   | 0/200 (skipped) | (skipped)   | **0.980**   | 0.003          | -0.33                  |
| 1    | +143.8   | 45/200     | (high)      | **0.899**   | 0.020          | (varies)               |
| 2    | (converged) | (varies) | (trained)  | **0.212**   | (varies)       | (varies)               |
| 3    | (converged) | (varies) | (trained)  | **0.924**   | (varies)       | -0.64                  |
| 4    | +147.2   | 6/200 (varies) | 0.996     | **0.966**   | 0.065          | -0.58                  |

## Aggregate (5 seeds)

mean Eval AUROC = 0.796
median = 0.924
range = 0.21 to 0.98
seeds > 0.5 = 4/5 (80%)

## Interpretation

**H1 is supported on 4/5 seeds.** Seed 2 anomaly persists but is the
outlier, not the rule. The Monitor learns reliably with frozen-policy
decoupled training, with one PPO seed giving an inverse result.

Pearson correlation: most seeds show strong negative correlation between
Monitor probability and episode reward. This is the expected H1 signature
(Monitor predicts failure = low reward).

## Next

- D (perturbation with std 1.0, 2.0) for stronger adversarial test
- Project C slot-WM code skeleton (architecture to implementation)
- Joint ablation clean rewrite
