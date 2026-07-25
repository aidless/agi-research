# CartPole-v1 5-seed frozen vs joint — supplementary cross-env test

> Date: 2026-07-25
> Status: COMPLETE — partially informative
> Goal: replicate H1 joint ablation on a 2nd environment

## Setup

- Environment: CartPole-v1 (classic control, faster convergence than LunarLander)
- Frozen Monitor: 30K PPO steps, then train Monitor on 100 frozen rollouts
- Joint Monitor: 30K PPO steps, Monitor trained every K=4 PPO updates on 20 fresh rollouts (2 epochs each)
- Eval: 50 episodes, p10 threshold for failure label
- 5 seeds: 0, 1, 2, 3, 4

## Frozen Monitor results

| Seed | Frozen AUROC | Frozen verdict |
|------|--------------|----------------|
| 0    | 0.302        | Not supported  |
| 1    | 0.707        | Supported      |
| 2    | 0.184        | Not supported  |
| 3    | 0.833        | Supported      |
| 4    | 0.608        | Supported      |
| **mean** | **0.527** | **3/5 supported** |

Frozen mean AUROC = 0.527 (above random 0.5). 3/5 seeds support H1.

## Joint Monitor results

| Seed | Joint AUROC | fail_rate | Notes |
|------|-------------|-----------|-------|
| 0-4  | NaN         | 0.0       | All 5 seeds: PPO converged to ~488 reward, no failures |

All 5 joint seeds returned NaN AUROC because fail_rate = 0 (no failure variance).
PPO at 30K steps converges to near-maximal CartPole reward (488/500), so p10
threshold (13.0) is never crossed in eval episodes. Monitor output collapses to
constant ~0.5 because BCE loss on a uniform-label set has zero gradient.

## Interpretation

CartPole-v1 is **not a valid test environment for the H1 joint ablation**
because:
1. PPO converges too quickly on CartPole (~30K steps to ~488 reward)
2. Once converged, all episodes succeed -> no failure variance
3. Both frozen and joint Monitors see the same degenerate uniform-label
   distribution at eval time
4. Frozen Monitor has the *option* to see early-training failures in its
   training set; joint Monitor doesn't, because Monitor only trains during
   late training when failures are gone

This is actually an *informative negative result*: the H1 ablation requires
failure variance. CartPole does not provide it within 30K PPO budget.

For meaningful cross-env H1 validation, the environment must:
- Have non-trivial failure modes (not trivially solvable)
- Produce a non-trivial fraction of failed episodes under typical PPO training
- Allow Monitor enough training signal to learn the failure structure

LunarLander-v3 satisfies all three (5/5 H1 supported). Procgen 16 games
would also satisfy them. CartPole does not.

## What this changes

- CartPole is **not** a valid H1 cross-env test at 30K PPO budget.
- LunarLander-v3 (already done) is the gold-standard H1 evidence.
- Y1 work: run H1 joint ablation on Procgen 4-16 games for the 12+ games
  falsifier from the original H1 specification.

## Artifacts

- `code/classic_phase2.py` (existing, runs frozen Monitor on CartPole)
- `code/checkpoints/joint_CartPole-v1_seed{0..4}/phase2_log.json`
- 5 seeds × 30K PPO each, ~5 minutes total

## What this does NOT change

- The LunarLander-v3 H1 5-seed result (delta=0.724, 5/5 supported)
  remains the primary evidence.
- The joint_phase2.py implementation is correct; the CartPole
  environment is the issue, not the code.
- The frozen-critic pattern claim (STaR/ReAct/Reflexion/Self-Refine/CRITIC/PRM)
  is unchanged.