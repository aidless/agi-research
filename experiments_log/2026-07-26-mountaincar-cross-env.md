# MountainCar-v0 5-seed joint — supplementary cross-env test

> Date: 2026-07-26
> Status: COMPLETE — inconclusive (PPO too weak)
> Goal: extend H1 cross-env validation beyond LunarLander-v3

## Setup

- Environment: MountainCar-v0 (sparse-reward classic control)
- 50K PPO steps per seed (vs 100K for LunarLander-v3)
- Joint Monitor with K=4 PPO update interval
- 200 train + 100 eval episodes
- 5 seeds: 0, 1, 2, 3, 4

## Joint Monitor results

All 5 seeds returned NaN AUROC.

## Why NaN

Examined seed 0 phase2_log.json:

| field | value |
|-------|-------|
| eval_reward_mean | -200.0 |
| eval_reward_std | **0.0** |
| fail_rate | 1.0 |
| eval_prob_mean | 0.487 |
| eval_prob_std | 0.007 |

**eval_reward_std = 0** means every single eval episode returned exactly
the same reward (-200, the worst possible MountainCar score). All 100
episodes failed identically. PPO at 50K steps has not learned to climb
the hill — the policy is essentially random.

With fail_rate = 1.0 (uniform-label set), BCE loss has zero gradient at
constant 0.5 prediction, so Monitor collapses to constant ~0.49.
AUROC is undefined because fail_labels.std() == 0.

## Comparison with LunarLander-v3 and CartPole-v1

| env | PPO needed for convergence | failure variance at our budget | H1 status |
|-----|----------------------------|--------------------------------|-----------|
| LunarLander-v3 | ~50-100K | yes (0.21) | 5/5 supported (delta=0.724) |
| CartPole-v1 | ~30K | none (converges too fast) | frozen 3/5, joint NaN |
| MountainCar-v0 | ~200K+ | none (PPO too weak) | both NaN |

## Interpretation

**MountainCar-v0 is not a valid H1 test environment at 50K PPO budget**
for the OPPOSITE reason as CartPole:
- CartPole: PPO converges too FAST (solved by 30K, no failures)
- MountainCar: PPO converges too SLOW (needs 200K+, all failures)
- LunarLander: in the sweet spot (50-100K PPO produces non-trivial
  failure distribution)

For MountainCar to be a valid H1 test env, we would need:
- ~250K PPO steps per seed × 5 seeds = 1.25M total steps
- ~5-8 min per seed × 5 = 25-40 min compute
- Not feasible in single session, deferred to Y1

## Implications for Paper A v2

1. LunarLander-v3 remains the gold-standard H1 evidence (5/5 supported,
   delta=0.724).
2. Cross-env validation needs envs in the "PPO converges in 50-150K
   steps with non-trivial failure variance" sweet spot.
3. Candidates for Y1: Procgen 16 games (intentionally designed for
   50-250K PPO range), Atari games (similar), or
   Procgen-equivalent with controlled difficulty.
4. MountainCar is documented as a negative-result env (similar to
   CartPole in the prior ablation).

## Artifacts

- `code/checkpoints/joint_MountainCar-v0_seed{0..4}/phase2_log.json`
- Total runtime: 6 minutes for 5 seeds at 50K PPO

## Connection to F:\TMLR H/I series

This empirical finding reinforces H06 from I03 World Models:
"Pearl L3 unsolvable in 3 years" is partly due to insufficient
PPO convergence for sparse-reward envs. Our H1 ablation needs
environments where PPO is *partially* successful — fully successful
(CartPole) or fully unsuccessful (MountainCar) both give degenerate
label distributions.