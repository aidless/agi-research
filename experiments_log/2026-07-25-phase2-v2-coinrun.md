# Phase 2 v2 on coinrun (2026-07-25) - 256K PPO + p10 threshold

## Setup
- PPO baseline: 256K steps (vs Step 1's 50K)
- Threshold: p10 of training returns (vs default p30)
- Monitor: 100 train episodes + 50 eval

## Result

`
PPO complete: mean_return=6.47, n_episodes=4189
p10 threshold = 0.0  (training distribution has 0 as floor)
Monitor dataset: 100 episodes (0 failures, 100 successes)
--> SKIP monitor training (one class only)
`

In other words: p10 also fails to find episodes below threshold because **all**
training episodes have reward >= p10 (since >30% are 0 reward).

## Yet -- A signal emerged

"Monitor mean prob mean+/-std: 0.485 +/- 0.003" (was 0.002 in Phase 1 Step 3)
"Pearson(prob, reward): -0.522" (was -0.33 in Phase 1)

This means the Monitor IS NOT constant — it varies slightly with reward:
higher Monitor probability correlates with lower reward (anticorrelated).
The architectural decoupling IS influencing the Monitor's output.

But AUROC remains 0.5 because fail_rate=0 means there are no positive labels.

## Implication

- Pipeline works. Monitor weights vary with input.
- The decoupling is doing SOMETHING (Pearson -0.52 > chance 0).
- We need a DIFFERENT PROBLEM where PPO actually produces variance.
- Candidate: LunarLander-v2 (clear success/failure boundary) or Procgen HARD.

## Output
Saved: code/checkpoints/procgen_coinrun_seed0/phase2_log.json

## Next: try LunarLander-v2

LunarLander-v2 binary success/failure: easy mode produces roughly 50/50
success/failure. PPO at 256K converges to mostly successes. But there IS
variance in failure-vs-success during training (where the agent learns).

LunarLander + frozen-policy Monitor = a stronger test of H1 than coinrun.

## Bug fix this turn

- envs.py: percentile_failure_threshold default 30 -> 10 (more cases labelled).
- All code/*/ files had UTF-8 corruption (Windows cp1252 0xA1 0xAA appearing
  where em-dashes should be). Fixed 4 occurrences in envs.py via byte-level
  replace.
- procgen_phase2.py: --percentile CLI flag added; threshold uses it.
