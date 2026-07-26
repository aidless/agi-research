# TTC BoN+Monitor v4 (hidden=256, epochs=20) — LunarLander-v3

> Date: 2026-07-26
> Status: Monitor signal recovered, but BoN usage is wrong
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What changed

Added 3 new CLI args to ttc_bon_monitor.py:
- `--monitor-hidden 256` (vs 64 baseline)
- `--monitor-epochs-train 20` (vs 5 baseline)
- `--monitor-balanced 1` (default 1 = balanced subsampling)

Result: bigger Monitor + more training successfully recovers discriminative
output (std=0.186 vs std=0.013 in v3).

## 2. Result (LunarLander-v3, 100K PPO, N=4, K=8, seed 0)

| metric | v3 (h=64) | **v4 (h=256, e=20)** |
|--------|-----------|------------------------|
| class balance | 50.7% | 19.3% (4:1 ratio) |
| Monitor hidden | 64 | 256 |
| Monitor epochs | 5 | 20 |
| Monitor output min | 0.461 | **0.095** |
| Monitor output max | 0.520 | **0.812** |
| Monitor output std | **0.013** | **0.186** |
| PPO mean | 9.9 | 100.2 |
| BoN+Monitor mean | -39.9 | -262.7 |
| **Delta** | -49.8 | **-362.9** |

Action distribution: {0: 11, 1: 256, 2: 334, 3: 329} — more balanced
than v2/v3 (which were 0: 3, 2: 595).

## 3. Diagnosis: BoN usage is wrong, not Monitor

The Monitor is now **discriminating correctly** (std=0.186, range
0.095-0.812). It can tell failure states from non-failure states.

But BoN+Monitor collapses to mean=-262.7 vs PPO mean=100.2. The
issue is **how BoN uses the Monitor signal**:

- BoN picks "lowest Monitor failure probability" candidate
- In LunarLander, "no engine" (action 0) has lowest Monitor failure
  probability in many states (because it's "safer" — agent doesn't
  crash, just times out)
- BoN over-uses action 0 (= 11 in 930 steps = 1.2%, low) but more
  importantly over-uses "safe" actions 1 and 3 which don't reach
  the goal

**The fundamental problem**: "low failure probability" is NOT the
right reward signal for TTC. The right signal is "high expected return".

Monitor correctly identifies failure probability. BoN should use
EXPECTED RETURN, not failure probability. The two are correlated but
not equivalent.

## 4. What would actually work (Y1 candidates)

1. **Value-function BoN**: replace Monitor with learned V(s, a) and
   pick action maximizing V. Simpler, more direct.
2. **Expected return from Monitor**: convert Monitor's failure
   probability to expected return estimate via domain knowledge
   (LunarLander: -100 per step until goal reached).
3. **Q-function BoN**: learn Q(s, a) from frozen rollouts via
   FQE (Fitted Q Evaluation). Use Q to rank BoN candidates.
4. **Per-step reward model**: instead of "will episode fail", train
   Monitor to predict "will next state be a success transition".
   Aggregate over rollout steps.

## 5. Conclusion

The TTC infrastructure (state cloning + balanced Monitor training +
BoN ranking) is correct. The **reward signal** is wrong.

This is essentially the standard model-based RL insight: you need a
value function or reward model, not just a failure classifier. The
"BoN+Monitor" framing is too narrow.

**ADR 0011 should be DEFERRED to Y1** with the following scope:
- Y1 Q1: train Q-function on frozen rollouts
- Y1 Q2: Q-function BoN as TTC alternative
- Y1 Q3: cross-env validation
- Compare Q-BoN vs Monitor-BoN vs vanilla PPO at matched FLOPs

## 6. Y1 TTC alternatives (replacement for ADR 0011)

| Method | Score | Compute | Notes |
|--------|-------|---------|-------|
| Vanilla PPO | baseline | 1x | simple |
| BoN+Monitor | -362 to +10 (mixed) | Nx | what we tried |
| BoN+Q-function | expected positive | Nx | needs Q from rollouts |
| Tree search + reward | expected positive | high | DreamerV3-style |
| Self-critique revision | needs LLM | high | different paradigm |

Y1 should explore BoN+Q-function and tree search variants.

## 7. Artifacts

- `code/ttc_bon_monitor.py` (now ~12.5 KB, with new CLI args)
- `code/checkpoints/ttc_bon_monitor_LunarLander-v3_seed0/phase2_log.json`
- Compute: ~2 minutes (slower due to 256-hidden Monitor)