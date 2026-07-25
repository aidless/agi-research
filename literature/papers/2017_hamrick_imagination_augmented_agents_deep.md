# Imagination-Augmented Agents (Hamrick et al. 2017, DeepMind)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH** -- classic DeepMind paper; multiple follow-ups cite it
> One-line: Train RL agents whose policy is conditioned on rollouts from a
> learned environment model, making the policy imagine before acting.

---

## Problem the paper is solving

Standard model-free RL is sample-inefficient. Model-based RL (rollouts in
learned dynamics) was known to help but typically the imagined trajectories
were used for planning, not directly for policy learning.

Direct policy learning on imagined trajectories has been tried but policies
can overfit to model imperfections: they learn to exploit the model's
hallucinations, which fail to transfer to real env.

## Method

Two central ideas:

1. **Imagination-augmented policy**: the policy takes BOTH the current real state AND a
   set of K imagined rollouts (from a learned model), and produces the action. The
   rollouts are like additional "what-if" inputs to the policy.

2. **Iterative refinement**: the model is updated, then the policy against the model,
   then the rollout set against the policy, and so on. This is similar to Dyna-Q
   (Sutton 1991) but with deep networks.

Architecture:
- Model: deterministic next-state predictor (uses true state + action)
- Imagination: K parallel rollouts of length H
- Aggregator: combines the K imagined rollouts into a "summary" embedding
- Policy: takes (state, summary) -> action

## Empirical result

Beam-search-like improvements on Sokoban and a robotic pushing task; more sample-efficient
than DQN baselines. Specifically:
- Sokoban: solved levels with fewer real env interactions
- Push task: handled novel configurations

Compute cost: requires K=10 rollouts of length H=10 at each step; ~10x more compute than
DQN per step, but fewer total steps needed.

## Criticisms (specific)

1. **The policy learns to game the model**. When the model is wrong, the policy exploits
   the error. Iterative refinement helps but does not eliminate.

2. **K=10 imagined rollouts is an arbitrary choice**. The paper does not have principled
   guidance for K, H, or sampling distribution.

3. **The summarisation of rollouts is task-specific**. The aggregator architecture in
   Sokoban (which includes a learned attention over rollout content) doesn't transfer
   to push without redesign.

4. **No provision for monitoring failure of imagined rollouts** -- this is exactly
   Project A's gap.

## Connection to our program

This is **the direct conceptual predecessor of Project A**.

The paper shows that imagined rollouts HELP policy learning. Project A's Monitor
adds the next layer: when imagining, the agent should know which imagined rollouts
are realistic vs hallucinated.

Moreover, "Project A above Imagination-Augmented Agents" is a future direction:
- Hamrick et al trained policy on rollouts; we add failure-prediction on rollouts
- The Monitor above the model is the conceptual extension

We should cite this paper in Project A Related Work 2.1 (this paper is the conceptual parent).

## Confidence

HIGH. Multiple reproductions.

Re-verify:
- exact K=10 default
- exact H=10 default
- the specific Sokoban and pushing results

## Related papers

- Dyna-Q (Sutton 1991, "Dyna integration")
- Dreamer V1 (Hafner 2020)
- I2A (Weber 2017)
- Dreamer V3 (Hafner 2025)

## Status

- [x] cite in Project A Related Work 2.1 (conceptual parent)
- [x] directly motivates Project A's "monitor above the model's rollouts"
- [ ] cite in Project D (imagination is a planning ingredient)
