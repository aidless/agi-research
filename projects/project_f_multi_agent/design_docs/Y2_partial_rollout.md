# Y2 Project F Design Doc: Partial Rollout for Multi-Agent RL

> Date: 2026-07-29
> Author: Liu Zewen (with Codex agent)
> Status: Design doc (no code yet). For Y2 implementation.
> Inspiration: Kimi K3 (Moonshot AI, 2026) partial rollout scheme.

---

## 1. Background

### 1.1 Kimi K3 partial rollout (from tech report)

Kimi K3's post-training uses a **partial rollout** scheme for agentic
RL:

- During the rollout phase of each iteration, sample K completions
  for each of N prompts (active workload: N x K trajectories).
- Rather than waiting for all rollouts to terminate, the generation
  phase pauses as soon as a fraction `lambda in (0,1)` of trajectories
  complete (i.e., `lambda * N * K`).
- Paused rollouts are enqueued and prioritized for resumption at the
  start of the next iteration.
- Policy optimization follows in a separate step; tolerates extreme
  off-policy via per-token regularization.

**Result**: scales RL FLOPs efficiently even when individual
trajectories have highly variable lengths (long-horizon tasks take
much longer than short-horizon tasks).

### 1.2 Y0 Project F (current state)

Y0 Project F is at the "sketch" stage:
- PettingZoo Simple Spread v3 base + DMC architecture skeleton.
- Y1 multi-agent results:
  - DMC continuous real shaping: H5 REFUTED (1/5 positive, t=-2.53).
  - MADDPG v2 baseline: +7.7 vs random, p<0.001.
  - Y2 A+B+C work: 6-way experiment, 2 negatives + 1 tentative positive.

The current Y1 MA training is **synchronous**: each iteration waits
for all N agents to complete their rollouts. With heterogeneous agent
speeds (some agents may take 10x longer than others), this is
inefficient.

## 2. Design proposal: Y2 Project F with partial rollout

Adopt Kimi K3's partial rollout scheme for multi-agent RL:

```
At iteration t:
  1. Each agent n has a current trajectory (started at some
     previous iteration t - k_n).
  2. Active rollouts: N agents, each generating one trajectory.
  3. Wait until lambda * N agents have completed their trajectories
     (i.e., reached terminal state).
  4. Collect completed rollouts; pause incomplete ones.
  5. Run policy optimization on the completed rollouts.
  6. Resume paused rollouts at the start of the next iteration.
```

This requires per-agent **state caching**: an agent that pauses
mid-trajectory must remember its full state (observation, action
history, hidden state) for resumption.

## 3. Why this could help

### 3.1 Wall-time reduction
If the slowest agent takes 10x longer than the fastest, the
synchronous loop wastes 9x wall time waiting for it. Partial rollout
unlocks the slow agent to continue in the next iteration while
the fast agents' policy updates run.

### 3.2 Effective batch size
More rollouts can be completed per wall-time unit, increasing the
effective batch size for policy optimization.

### 3.3 Heterogeneous agent speeds
In PettingZoo Simple Spread, agents may have heterogeneous
"natural" trajectory lengths (some agents finish their task early,
others get stuck in cycles). Partial rollout handles this naturally.

## 4. Pre-registered hypothesis: H14

**H14**: A partial rollout scheme (lambda = 0.5) for multi-agent
RL on PettingZoo Simple Spread v3 achieves:
- Equal or better final performance (mean episode return) vs
  synchronous rollout.
- Lower wall-time to reach the same performance (e.g., 2x faster
  to reach 80% of synchronous peak).

**Decision rule** (pre-registered before any data collection):
- VALIDATED if partial-rollout final return is within 0.05 of
  synchronous AND wall-time is at least 1.5x faster.
- REFUTED if partial-rollout final return is >= 0.10 worse than
  synchronous.
- INCONCLUSIVE otherwise.

**Pre-registered sample size**: n=5 seeds, 1000 iterations per
seed. Compute estimate: ~6 hours per seed on CPU. Total: ~30 hours
for n=5. **This is the most expensive hypothesis in the Archimedes
program**.

**Pre-registered negative control**: synchronous rollout (lambda=1.0).

## 5. Implementation outline

```python
class PartialRolloutTrainer:
    def __init__(self, n_agents, lambda_=0.5, max_iterations=1000):
        self.n_agents = n_agents
        self.lambda_ = lambda_  # fraction to complete per iter
        self.max_iterations = max_iterations
        # Per-agent state cache for paused rollouts.
        self.agent_states = [None] * n_agents

    def step(self, agents, env):
        # Run rollouts until lambda * N complete.
        target_complete = int(self.lambda_ * self.n_agents)
        completed = []
        paused = []
        for i, agent in enumerate(agents):
            if self.agent_states[i] is not None:
                # Resume from cache.
                state = self.agent_states[i]
            else:
                state = env.reset(agent=i)
            done = False
            trajectory = []
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(agent=i, action=action)
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
                if self._check_global_stop(target_complete, len(completed)):
                    # Stop this rollout; cache state for next iter.
                    self.agent_states[i] = state
                    paused.append((i, trajectory))
                    break
            else:
                # Completed.
                completed.append((i, trajectory))
                self.agent_states[i] = None
        return completed, paused
```

The `_check_global_stop` method checks if `lambda * N` agents have
completed; if so, it pauses the rest.

## 6. What is novel vs existing work

### 6.1 vs Kimi K3
Kimi K3 uses partial rollout for **single-agent** LLM training.
Our design adapts it for **multi-agent** RL, where state caching
must handle concurrent agent-environment interactions (vs single
sequential text generation).

### 6.2 vs prior multi-agent RL
Standard multi-agent RL (MADDPG, QMIX, COMA) is synchronous. Async
methods (A3C, GA3C) exist for single-agent, but per-agent
state caching for multi-agent envs is less developed.

### 6.3 vs PettingZoo Async
PettingZoo supports `parallel` mode where all agents act
simultaneously, but this is at the env level (per-step), not at the
rollout level (per-trajectory). Our partial rollout is at the
rollout level.

## 7. Risk and honest framing

### 7.1 Risks
- **Stale policy**: partial rollout introduces off-policy data;
  the policy may diverge. Per-token regularization (Kimi K3) or
  importance sampling (IMPALA) can help.
- **State caching complexity**: per-agent state caches must be
  thread-safe and consistent across pauses.
- **Compute**: 30 hours for n=5 is the most expensive hypothesis
  in the Archimedes program. If budget is limited, this may be
  infeasible.
- **Off-policy correction is non-trivial**: PPO is on-policy; with
  partial rollout, the data is off-policy. Need careful PPO clipping
  or PPO-with-IS variants.

### 7.2 Honest framing
This is a **design doc** with a pre-registered hypothesis (H14).
The implementation can begin when Y2 funding/time allows. The
H14 result (when run) will be the test of whether partial rollout
helps multi-agent RL.

If H14 refutes (partial rollout is significantly worse), the
conclusion is: synchronous rollout is the right baseline for
multi-agent RL at our scale; partial rollout is only useful at
frontier-model scale (Kimi K3 has 10K+ rollouts per iteration; we
typically have 5-20).

## 8. Relation to existing Archimedes work

### 8.1 Project F Y1
Current Y1 multi-agent training is synchronous. Partial rollout
is a Y2 direction. If H14 validates, we can apply it to all future
Y2/Y3 multi-agent experiments.

### 8.2 Project A (Y1.3, Y2 H13)
Single-agent Project A training is also synchronous. If H14
validates for multi-agent, we can also apply partial rollout to
single-agent Y1.3 (less benefit since single-agent envs have less
trajectory-length variance).

## 9. Next step

For Y2 implementation (not in current session):
1. Implement `PartialRolloutTrainer` per the outline above.
2. Add H14 pre-registration file with hard decision rule.
3. Run n=5 seeds at 1000 iterations per seed on PettingZoo
   Simple Spread v3.
4. Compare final return and wall-time-to-80%-peak vs synchronous.
5. If H14 validates, write a Y2 paper on the contribution.

This design doc is sufficient as a Y2 roadmap; the actual
implementation can begin when Y2 funding / time allows.

---

*Design doc prepared 2026-07-29 by Codex agent. Inspiration: Kimi K3
partial rollout scheme [46]. For Y2 Project F implementation.*