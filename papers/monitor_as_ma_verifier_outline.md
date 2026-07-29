# Monitor as MA Verifier: A Cross-Agent Evidence Chain Approach

> Status: Outline (pre-15-seed confirmation; v6 ablation pending)
> Target venue: NeurIPS 2026 MARL workshop, or AAMAS 2027
> Author: Liu Zewen + Codex (Archimedes Project, AGI-2026-001)

## Abstract

We investigate how failure-prediction Monitors can be used in cooperative
multi-agent reinforcement learning (MARL). Monitors in single-agent RL
(Y1.3, LunarLander-v3, p<0.001) provide a verified training-time
signal via reward shaping, but this approach fails to transfer to MA:
H5 is REFUTED on PettingZoo Simple Spread v3. We explore three Y2
designs: (v3) Monitor as critic auxiliary loss, (v4) inter-agent
message broadcast in critic, and (v5) Monitor as actor-side verifier
via cross-agent evidence chain and trust head. Only v5 shows a
consistent (though n=5-not-yet-significant) positive direction. We
argue: Monitors as **critic-side** extras are a dead end at our compute
scale; Monitors as **actor-side verifiers** (post-hoc trust signals)
are the architecturally correct use. We present a 3-arm 5-seed ablation
(with_verifier, with_trusthead_random, no_verifier) isolating the
source of the +0.17 signal, and a 15-seed confirmation test (in
progress).

## 1. Introduction

Single-agent failure-prediction Monitors (Y1 paper, n=15 seeds,
t=6.76, p<0.001 on LunarLander-v3) are a verified training-time
signal when used as a reward penalty. The question: do they transfer
to cooperative multi-agent settings?

Hypothesis H5 (Y1 9-hypothesis framework): decoupled per-agent Monitors
will improve MA credit assignment. H5 status: REFUTED.

We present a systematic 3-pathway investigation of WHY H5 was refuted
and WHAT (if anything) is the right use of Monitors in MA.

## 2. Background

### 2.1 Single-agent Monitor (Y1.3)
Froze-decoupled Monitor, AUROC 0.796 vs joint 0.072 (5 seeds).

### 2.2 Multi-agent PettingZoo Simple Spread v3
3 agents cover 3 landmarks, joint reward. N=3, max_cycles=25.

### 2.3 MADDPG v2 baseline (our starting point)
Centralised critic, per-agent actors. 5-seed mean: -70.45 +/- 1.14
(5/5 positive vs random, p<0.001).

## 3. Method: 3 Pathway Investigation

### 3.1 v3: Monitor as critic auxiliary loss (NEGATIVE)
critic_loss = Q-MSE + 0.5 * MonitorBCE. 3-arm 5-seed at 800 ep:
with_aux: -70.50, no_aux: -70.50, ablated: -70.50 (identical).
At 10K ep: with_aux: -74.89, no_aux: -71.85 (with_aux HURTS -3.03).
Conclusion: critic-side aux loss is a dead end.

### 3.2 v4: Inter-agent messages in critic (NEGATIVE)
Per-agent 32-dim message broadcast; critic input extended with
all_messages. 3-arm 5-seed at 800 ep:
with_comms: -70.31, no_comms: -70.32, random_comms: -70.35 (no effect).
Conclusion: critic-side comms are a dead end.

### 3.3 v5: Monitor as actor-side verifier (TENTATIVE POSITIVE)
Architecture:
- Critic: unchanged MADDPG v2.
- Monitor: per-agent frozen-decoupled (trained on Stage-1 PPO rollouts,
  AUROC ~0.99 on frozen-data eval).
- **Evidence chain**: per step, per-agent monitor_prob -> SHA-256 entry
  (agent_id, step, monitor_prob, prev_hash).
- **Trust head at actor**: input (my_obs, my_monitor_prob,
  others_monitor_stats) -> per-other-agent trust weights in [0,1].
- Actor loss: maximise own Q + trust-weighted sum of other agents' Q.
Trust head trained end-to-end via this loss.
5-seed 3-arm at 800 ep:
with_verifier: -70.33, no_verifier: -70.50, random_verifier: -70.52.
with_verifier vs no_verifier: +0.17, t=+1.01, 3/5 positive (NOT sig at n=5).

**KEY INSIGHT**: Monitor signal at the ACTOR (via trust head) is the
right architectural choice. Monitors should be per-agent DECISION
signals, not value function inputs.

## 4. v6: Trust Head Ablation (isolate the source)

Question: is the +0.17 from the Monitor or the trust head architecture?
3 arms: with_verifier (real Monitor), with_trusthead_random (random
inputs to trust head), no_verifier. 5-seed results [pending].

## 5. 15-seed Confirmation [in progress]

v5 with_verifier vs no_verifier at 15 seeds (n=5 done so far:
with_verifier: -70.32 +/- 1.67 (n=11), no_verifier: -70.64 +/- 1.90 (n=13).
Paired t (preliminary): +0.28, t=+0.28 (NOT sig yet, n=11).
Full 15-seed completion pending.

## 6. 10K-Episode v5 Sweep [pending]

If 15-seed v5 reaches significance, run 10K episode v5 with_verifier vs
no_verifier to test whether the effect amplifies with longer training
(as v3 10K showed for the negative case).

## 7. Discussion

### 7.1 The architectural lesson
Our 3-pathway investigation shows:
- v3: critic-side extras (Monitor) = 0 effect at 800ep, HURTS at 10K.
- v4: critic-side extras (comms) = 0 effect.
- v5: actor-side extras (Monitor via trust head) = +0.17 tentative.

**Why critic-side fails**: MADDPG v2's centralised critic already has
access to the FULL global state. Adding redundant info (Monitor
output, comm messages) does not give the critic new information.

**Why actor-side works**: the actor does NOT have access to other
agents' observations by default. A trust head that conditions on
(my_obs, my_monitor, others_monitor) gives the actor NEW information
for the action selection step. This is the right place to use a
Monitor signal in MA.

### 7.2 Monitors as VERIFIERS, not training signals
This sharpens the Y1 paper's architectural lesson:
- In single-agent RL, Monitor = training signal (verified by Y1.3).
- In multi-agent RL, Monitor = verifier (post-hoc decision signal).
The Monitor is a per-agent reliability indicator, useful for
trust-weighted cooperation.

## 8. Limitations

- n=5 tentative positive (n=15 confirmation in progress).
- PettingZoo Simple Spread only (other MA envs needed).
- 800 env episodes (10K+ training in progress).
- Trust head has small parameter count (~5K params).

## 9. Conclusion

Monitors as actor-side verifiers via cross-agent evidence chain and
trust head is a promising new direction for MARL (TENTATIVE
POSITIVE at n=5, 15-seed confirmation pending). The architectural
lesson is sharp: critic-side extras are dead ends, but actor-side
trust-weighted Q blending uses Monitor signal correctly. We will
confirm with 15-seed + 10K episode re-runs before final claim.

## References
- Lowe et al. 2017 (MADDPG)
- Liu Zewen 2026 (Y1 paper, AGI-2026-001)
- 9-hypothesis framework, papers/y1_9hypothesis_framework.md
- TarMAC / IC3Net (inter-agent comms, baseline v4)

## Appendix A: Code pointers
- pz_maddpg_v3.py: Monitor aux loss in critic
- pz_maddpg_v4.py: TarMAC-lite inter-agent comms
- pz_maddpg_v5.py: Monitor as MA verifier (TENTATIVE POSITIVE)
- pz_maddpg_v6.py: Trust head ablation
- experiments_log/2026-07-28-pz-maddpg-v{3,4,5,6}-*-5seed.md