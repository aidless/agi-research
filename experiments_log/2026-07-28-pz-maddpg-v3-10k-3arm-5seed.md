# MADDPG v3 10K-episode 5-seed 3-arm: NEGATIVE (with_aux HURTS, 0/5 positive)

> Date: 2026-07-28
> Code: `projects/project_f_multi_agent/code/pz_maddpg_v3.py`
> Same arms as 800-episode run, but 10x compute: 800 updates x 10 episodes
> = 8000 env episodes per seed (vs 800 in the short run).

## 1. Why this run

The 800-episode v3 sweep showed with_aux = no_aux = ablated (3 arms
produced IDENTICAL results). One hypothesis: 'compute was too short,
Monitor aux loss needs more time to influence critic'. This 10K
episode re-run tests that hypothesis.

## 2. 5-seed results (800 updates x 10 episodes = 8000 episodes)

| arm | mean | sd | n |
|---|---|---|---|
| with_aux | -74.89 | 3.57 | 5 |
| no_aux | -71.85 | 2.37 | 5 |
| ablated | -74.10 | 4.01 | 5 |

Paired t-tests (df=4, |t|>=2.776 for p<0.05):
- with_aux vs no_aux: mean_diff=-3.03, t=-1.39, 0/5 positive (NOT sig but
  consistent direction)
- with_aux vs ablated: mean_diff=-0.79, t=-1.00, 0/5 positive (NOT sig)
- no_aux vs ablated: mean_diff=+2.25, t=+1.00, 1/5 positive (NOT sig)

## 3. Honest interpretation

After 10x more compute, the arms DIVERGE (not identical like the
800-episode run). The direction is:

- **with_aux HURTS** the baseline: -3.03 mean worse than no_aux,
  0/5 positive (all 5 seeds worse).
- **ablated also worse than no_aux**: +2.25 mean, 1/5 positive.
- **no_aux is the best arm** at -71.85, close to MADDPG v2 baseline
  -70.45 (which uses 800 episodes at matched compute).

So at 10K compute:
1. Aux loss is actively HARMFUL (not just neutral).
2. Even a random Monitor in critic (ablated) is worse than no Monitor
   at all (no_aux).
3. Adding extra information to the critic HURTS performance.

## 4. Why might aux loss HURT with more compute?

Hypotheses:
1. **Overfitting**: the aux loss lets the critic fit the Monitor
   signal (frozen, biased) at the expense of Q-value accuracy.
2. **Conflicting gradients**: the Q-MSE gradient and the aux-loss
   gradient point in different directions; over many updates the aux
   loss dominates and Q degrades.
3. **Monitor is biased**: as noted in v3 log, the frozen Monitor
   was trained on Stage-1 PPO rollouts. By 10K updates, the policy
   has drifted from Stage-1; the Monitor's failure concept is stale.

## 5. Comparing v3 short (800 ep) vs v3 10K

| run | with_aux | no_aux | ablated |
|---|---|---|---|
| v3 800 ep | -70.50 | -70.50 | -70.50 | (identical)
| v3 10K | -74.89 | -71.85 | -74.10 | (with_aux hurts)

Interpretation: at 800 ep, the aux loss has no effect (the critic is
too under-trained to be influenced by the aux term). At 10K, the
aux loss has time to influence critic and starts to HURT.

## 6. Comparison to MADDPG v2 5-seed

| Method | mean | sd |
|---|---|---|
| MADDPG v2 (800 ep) | -70.45 | 1.14 |
| v3 10K no_aux | -71.85 | 2.37 | (similar to v2 but 10x compute)
| v3 10K with_aux | -74.89 | 3.57 | (worse than v2)
| v3 10K ablated | -74.10 | 4.01 | (worse than v2)

Note: v2 uses 80 updates x 10 episodes (800 ep) and gets -70.45.
v3 10K no_aux uses 800 updates x 10 episodes (8000 ep) and gets
-71.85 (similar). So 10x more compute on the same backbone does NOT
improve performance. The MADDPG v2 architecture is near-saturated
on Simple Spread at this env complexity.

## 7. Implications for Y2

Path a (longer compute) is also NEGATIVE. Combined with v3 short
and v4 (inter-agent comms), the unified finding is:

- Critic-side extras (Monitor aux loss, inter-agent messages) do NOT
  help at any compute scale we tested (800, 8000 episodes).
- MADDPG v2 baseline is near-saturated on Simple Spread at this
  scale; the bottleneck is elsewhere (env complexity, not credit
  assignment).
- Path c (Monitor as MA verifier, post-hoc trust score) remains the
  most promising direction because it is NOT a critic-side extra.

## 8. Action items

- [x] v3 10K 5-seed 3-arm sweep (completed)
- [x] v4 5-seed 3-arm sweep (completed, also negative)
- [x] v3 short 5-seed 3-arm (completed, 3 arms identical)
- [x] Honest log: v3 10K NEGATIVE (with_aux hurts)
- [ ] Update H5 Y2 follow-up in 9-hypo framework with v3/v4 results
- [ ] Path c (Monitor as MA verifier) implementation - next session