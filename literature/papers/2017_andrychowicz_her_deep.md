# HER - Hindsight Experience Replay (Andrychowicz et al. 2017)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: When an episode fails to reach its goal, treat the achieved
> final state AS IF it were the intended goal. This converts failures
> into successes for off-policy replay.

---

## Problem

RL on goal-conditioned tasks (e.g., robot arm pushing a block to a
specific position) is brutally hard when rewards are sparse. A naive
agent that explores randomly never sees positive reward, so never
learns.

## Method

Standard experience replay: store (s, a, r, s', goal, achieved_goal).
Standard relabeling: replay (s, a, r, s', goal) and update policy.

HER trick: ALSO replay with a **retrofitted goal**. Specifically:
- "future" strategy: pick a goal that was achieved LATER in this
  trajectory. Re-label that (s, a) tuple as if THAT was the goal
  we were trying to achieve.
- The reward signal becomes positive for that tuple.
- Q-function learns: with appropriate hindsight, the trajectory
  taught us something.

This requires the env to be goal-conditioned (state can be split into
achieved vs desired parts).

## Empirical result

HER + DDPG on FetchPush / Slide / Pick-and-Place (MuJoCo robot):
- solves tasks with binary success reward (sparse)
- compared to vanilla DDPG: orders-of-magnitude sample efficiency
  improvement

HER + SAC: SOTA manipulation benchmarks on Fetch and dexterous hand
tasks at the time.

## Criticisms

1. **Only works for goal-conditioned tasks**. If env doesn't have a
   goal space, HER is irrelevant.

2. **Goal-space design matters**. You need to know which subset of state
   is "goal". For a Fetch arm, position of object; for an Atari game,
   this is less clear.

3. **Doesn't actually solve sparse-reward without a curriculum**. You
   still need exploration to discover variety; HER then retrospectively
   teaches. But variety has to come from somewhere.

4. **HER is essentially a data-augmentation trick, not a principled
   advance**. It does not introduce new structure; it just changes
   labels of existing data.

5. **May interfere with policy in subtle ways**. The Q-function learns
   "this state is good IF this is the goal"; at deployment you specify
   the actual goal. The conditional nature can cause optimisation
   pitfalls.

## Connection to our program

HER is a **direct inspiration for Project A's failure-label design**:

- Standard Project A monitor is trained on (state_history, fail_label)
- HER teaches us: an episode that "failed" is a positive example for
  what failure looks like
- We can retro-generate **counterfactuals**: in the same trajectory,
  find sub-episodes that DID succeed at something similar; treat these
  as positive examples for that sub-task
- This lifts our monitor's data efficiency

Practical version: our `envs.py:is_failure_episode` heuristic can be
augmented with hindsight relabeling: episodes that completed the task
(but got a low reward for other reasons) become positive examples.
This is directly HER-style.

## Related papers

- DDPG (Lillicrap 2015) - HER's basic algorithm partner
- SAC + HER (2018) - state of the art manipulation at the time
- HER + curriculum (OpenAI 2018)
- RHER (relabeled HER, 2019)

## Status

- [x] cite in Project A monitor label design
- [ ] incorporate hindsight relabeling into `code/envs.py` (Phase 2)
