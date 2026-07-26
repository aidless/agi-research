# Phase 2 Orchestrator: Self-aware agent with Monitor gating — NEGATIVE

> Date: 2026-07-26
> Status: COMPLETE — gating strategy wrong for LunarLander
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. Question

Can a self-aware agent (using its own Monitor prediction to gate its
own actions) improve over vanilla PPO? This is Phase 2 of the
AGI roadmap: end-to-end self-awareness.

## 2. Method

`code/orchestrator.py` (9277 bytes):

1. Train PPO on LunarLander (100K steps)
2. Train SlotMonitor (AUROC 0.989 from previous breakthrough)
3. Eval ungated PPO: 30 episodes, standard sampling
4. Eval gated: at each step, compute SlotMonitor probability. If
   prob > 0.5, override PPO's action with action 0 (do nothing).

Gating rule: "if I think I'm going to fail, do nothing" — a simple
conservative policy.

## 3. Results (LunarLander-v3, seed 0, 100K PPO, gate=0.5)

| metric | value |
|---|---|
| Ungated PPO mean | 36.2 +/- 50.8 |
| Gated (Monitor-guided) mean | -156.2 +/- 73.6 |
| **Delta** | **-192.3** |
| Gates triggered total | 1754 (avg 58.5/episode) |
| % steps gated | ~29% |

29% of actions were overridden to "do nothing". This causes fuel
exhaustion and timeout, leading to strongly negative reward.

## 4. Diagnosis: gating strategy is wrong for LunarLander

LunarLander requires active control — agent must thrust to maintain
altitude and reach the landing pad. "Do nothing" (action 0) is
catastrophic in this environment.

The Monitor correctly identifies "this state is dangerous" (high
failure probability), but the response "stop acting" is the wrong
one. The correct response would be something like:
- "Use minimal but active thrust" (action 1 or 3, depending on orientation)
- "Increase rotation control" if angle is wrong
- "Reorient toward landing pad" if drifting

We don't have these intelligent fallbacks. Our gating is naive:
"predict failure → do nothing" — wrong for active-control envs.

## 5. Lessons for self-aware agent design

1. **Gating action must be env-specific**: "do nothing" only works
   for envs where inaction is safe (CartPole might crash but quickly,
   Atari games where losing is OK)
2. **LunarLander needs "smart fallback", not "no action"**: a more
   sophisticated policy that knows which direction to thrust
3. **Better metric than failure probability**: use Q-function
   (Project A Q-BoN) to pick the BEST action, not "do nothing"
4. **Monitor should inform, not override**: use Monitor as an
   auxiliary signal combined with PPO, not as a hard gate

## 6. What would actually work

For LunarLander self-aware agent:
- Train Q-function on frozen rollouts (Q-BoN v5)
- At each step: PPO proposes action; if Monitor prob > threshold,
  REPLACE PPO's action with argmax_Q(s, a)
- This way Monitor triggers "more careful" behavior (Q picks safer
  action) rather than "no action"

Combine with CQL for the Q-function (from v6), this could work.

But that's Phase 2.5 — combine Monitor + Q-BoN + CQL.

For Y1: redesign gating to use Q instead of action 0. This is the
natural next step in the AGI roadmap.

## 7. Cumulative AGI roadmap progress

| phase | status |
|---|---|
| 1.1 A+C integration (slot-Monitor) | ✅ done, AUROC 0.989 |
| 1.2 C dynamics (Transformer dynamics) | not started |
| 1.3 D interface (LLM-as-type) | not started |
| 1.4 E verifier (LTL) | not started |
| 1.5 Full integration | not started |
| 2.1 Self-improvement loop (this attempt) | ❌ negative — gating wrong |

## 8. Artifacts

- `code/orchestrator.py` (9277 bytes, NEW)
- `code/checkpoints/orchestrator_LunarLander-v3_seed0/phase2_log.json`
- Compute: ~2.7 minutes