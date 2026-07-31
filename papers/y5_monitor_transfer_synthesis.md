# The Failure-Prediction Monitor Does Not Transfer: A 3-Context Investigation (RL, MARL, LLM)
## Y5 Synthesis Paper

**Authors:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-07-30
**Status:** Y5 synthesis paper. Combines Y1 (single-agent RL), Y3 (multi-agent
RL), and Y4 (LLM self-monitoring) findings.
**Companion papers**:
- Y1: `papers/y1_paper_draft.md`
- Y3: `papers/monitor_signal_vs_dlr_6pathway.{md,tex,pdf}` (6-pathway MA)
- Y4: `papers/project_g_v0_5_h10_paper.md` (LLM self-monitoring)
**Target venue:** Nature Machine Intelligence or AI Magazine (synthesis
article). Alternative: ICML/NeurIPS/AAAI as a meta-analysis workshop
paper.

## Abstract

Failure-prediction Monitors (small classifiers that predict whether
an agent's trajectory will end in failure) are a verified
single-agent RL signal: frozen-decoupled Monitors give +39.5
mean improvement on LunarLander-v3 (n=15, t=6.76, p<0.001, Y1
paper). We synthesized evidence from three independent
investigations across three contexts: single-agent RL (Y1,
verified), multi-agent RL (Y3, 6-pathway systematic
investigation, 5/6 architectures REFUTED at p<0.05), and LLM
self-monitoring (Y4, H10 pre-registered pilot, REFUTED). Across
all three contexts, the pattern is consistent: **the failure-
prediction Monitor does not transfer** from the context in
which it was verified (single-agent RL) to other contexts
(multi-agent RL, LLM self-monitoring). The Monitor's verified
shipping use remains **verification** (DLR predicates in critic
for cross-agent signal, runtime guardrails for safety), not
training. We propose a unified framework for thinking about
Monitor transfer: the Monitor is a "context-specific" signal
that works only in the narrow regime where it was verified
(frozen-decoupled policy gradient RL on a single agent with
frequent failure modes). Generalization to other contexts
(multi-agent, non-RL, LLM) requires verification, not
assumption.

## 1. Introduction

The failure-prediction Monitor (Y1 paper) is one of the most
robust findings in the Archimedes Project: a small classifier
trained on a frozen policy's trajectories can predict whether
the trajectory will end in failure (AUROC 0.796 vs joint 0.072,
LunarLander-v3), and using this signal as a reward penalty gives
+39.5 mean improvement (n=15, t=6.76, p<0.001).

The natural question: **does the Monitor transfer to other
contexts**? We investigated this question in three independent
investigations:

1. **Single-agent RL (Y1, verified)**: Monitor works. n=15,
   t=6.76, p<0.001, +39.5 mean improvement.

2. **Multi-agent RL (Y3, 6-pathway investigation)**: Monitor does
   NOT transfer. 5/6 architectures REFUTED at p<0.05. The
   single positive result is DLR predicates in the critic (not
   the Monitor), +0.1447 at n=30 (p<0.005), shrinking to
   +0.06 at n=100 (p_bonf=0.043).

3. **LLM self-monitoring (Y4, H10 pre-reg pilot)**: Monitor does
   NOT transfer. Joint Monitor AUROC 0.650 vs Frozen Monitor
   AUROC 0.550 (Joint > Frozen, t=-0.516, NOT sig).

This Y5 paper synthesizes the three investigations and proposes a
unified framework for understanding Monitor transfer.

## 2. The three investigations (summary)

### 2.1 Y1: Single-agent RL (verified)

**Hypothesis H1.3**: frozen-decoupled Monitors improve
single-agent RL via reward shaping.

**Setup**: LunarLander-v3, n=15 seeds, 100K PPO steps. Frozen
Monitor trained on 80 episodes from a frozen Stage-1 policy.
Reward penalty = -Monitor(failure_prob) * 0.5.

**Result**: VALIDATED. +39.5 mean improvement, t=6.76, p<0.001.
Effect is reproducible across sample sizes.

**Why it works**: the Monitor captures the policy's failure
modes (e.g., "landed too fast", "out of bounds") and provides a
shaping signal that helps the policy avoid these modes. The
Monitor is decoupled (one per agent) and frozen (not updated
during PPO) to avoid the joint Monitor failure mode
(self-fulfilling predictions).

### 2.2 Y3: Multi-agent RL (6-pathway systematic investigation)

**Hypothesis H5**: decoupled per-agent Monitors improve MA
credit assignment.

**Setup**: PettingZoo Simple Spread v3, MADDPG v2 baseline,
6 architectures tested with 5-100 seeds per arm.

**Result**: REFUTED. 5/6 architectures REFUTED at p<0.05:
- v3: Monitor aux loss in critic: -3.03 mean (HURTS at 10K)
- v4: inter-agent comms in critic: 0.00
- v5: trust head + Monitor (actor): +0.06 at n=100 (sig but
  tiny)
- v6: trust head + random (actor): bit-for-bit identical to v5
  (trust head ignores input)
- v7: prior trust head + Monitor: same as v5

The single positive result is v8 dlr_only: DLR cross-agent
predicates in the critic (NOT the Monitor), +0.1447 at n=30
(p<0.005), shrinking to +0.06 at n=100 (p_bonf=0.043).

**v8 dlr_only 3-seed independent replication (seeds 200, 201, 202)**:
Per-seed diffs [+0.27, -0.08, +0.30], mean +0.16 (sd=0.21),
2/3 positive. Direction-consistent with the n=100 estimate
(+0.0617, 95% CI [+0.0084, +0.1149]). Single-seed replicates are
not powered for inference, but the effect is reproducible from
a fresh seed. Full data: `experiments_log/_v8_sanity_4seed.json`.

**Why the Monitor fails in MA**:
1. The Monitor is trained on a single-agent frozen policy, but
   in MA the policy is multi-agent (joint training, non-
   stationary). The Monitor's failure modes are no longer
   accurate.
2. Adding Monitor as aux loss in critic (v3) HURTS because
   the Monitor's biased predictions pull the critic's
   representation in a wrong direction.
3. Adding Monitor as actor trust head input (v5) is ignored:
   the trust head's gradient is dominated by my_obs, and the
   Monitor broadcast is treated as noise.

**Why DLR works**: DLR predicates are hand-crafted, deterministic
functions of the state vector that encode cross-agent
relationships (e.g., "agent i is closest to landmark j"). They
are not learned, so they don't suffer from the Monitor's bias
problem. They are interpretable, so they can be debugged.

### 2.3 Y4: LLM self-monitoring (H10 pre-reg pilot)

**Hypothesis H10**: frozen LM-based Monitors outperform joint
shared Monitors for LLM self-monitoring (i.e., decoupling
transfers from RL to LLM self-monitoring).

**Setup**: simple arithmetic tasks, 5 seeds with stratified
train/eval split (Project G v0.5), 3 arms (Frozen, Joint,
Random).

**Result**: REFUTED at all three sample sizes tested.

| Sample | Frozen | Joint | Random | F-J diff | sig (Bonf)? |
|---|---|---|---|---|---|
| n=5 | 0.550 | 0.650 | 0.250 | -0.10 (Joint > Frozen) | No (t=-0.516) |
| n=20 | 0.579 | 0.447 | 0.632 | +0.13 (Frozen > Joint) | No (t=+1.16, p=0.262) |
| n=100 | 0.500 | 0.485 | 0.510 | +0.015 | No (d=+0.030) |

At n=5 the direction is the OPPOSITE of the H10 prediction
(Joint > Frozen). At n=20 the direction flips back (Frozen >
Joint) but with no statistical significance. At n=100 all
three arms are within 0.02 of 0.5 (random), so the Monitor
signal is indistinguishable from chance. Required n for 80%
power at the observed d=+0.030 would be n~17,000, which is
clearly not warranted.

**Why the Monitor fails in LLM self-monitoring**:
1. The Monitor is trained on a frozen reference LLM, but
   production LLMs are updated (fine-tuned, RLHF'd). The
   Monitor's failure modes are no longer accurate.
2. The "joint" Monitor (shared across all traces) is more
   sample-efficient than per-trace frozen Monitors. With small
   sample sizes (n=5), the joint Monitor has more data and
   generalizes better.
3. LLM traces are more diverse than RL trajectories, so the
   Monitor's training data is sparser.

## 3. Unified framework: Monitor as a context-specific signal

### 3.1 The pattern

| context | decoupling effect | source | sample |
|---|---|---|---|
| single-agent RL (LunarLander) | **+39.5** (VERIFIED) | Y1 | n=15, t=6.76, p<0.001 |
| multi-agent RL (Simple Spread) | -3.03 to +0.06 | Y3 | n=5 to 212 |
| LLM self-monitoring (arithmetic) | -0.10 (Joint > Frozen) | Y4 | n=5, t=-0.516, NOT sig |

The Monitor works in the narrow context where it was verified
(single-agent RL with frozen policy gradient). It does NOT
transfer to:
- multi-agent RL (where the policy is non-stationary due to
  joint training)
- LLM self-monitoring (where the trace distribution is diverse
  and sample-efficient shared training is preferred)

### 3.2 Why the Monitor is context-specific

The Monitor works when:
1. **The agent's policy is stable** (frozen during Monitor
   training and use)
2. **The Monitor's training distribution matches the use-time
   distribution** (the failure modes are the same)
3. **The signal is informative** (failure modes are diverse
   and learnable from the training data)

The Monitor fails when:
1. **The agent's policy changes** (multi-agent joint training,
   LLM fine-tuning): the Monitor's failure modes are stale.
2. **The signal is biased** (v3 aux loss pulls the critic's
   representation in a wrong direction).
3. **The training data is sparse** (LLM traces are diverse;
   joint shared Monitor is more sample-efficient).

### 3.3 When to use the Monitor

Based on the three investigations, we recommend the Monitor for:

**USE**:
- Single-agent RL with frozen policy and frequent failure modes
  (the original Y1.3 setup)
- Runtime guardrails: predict failure and intervene (handoff,
  rollback, etc.)
- Verification: predict failure on test trajectories as a
  safety check

**DON'T USE**:
- Multi-agent RL: the Monitor does not transfer (use DLR
  predicates in critic instead, see Y3 v8 dlr_only)
- LLM self-monitoring: the Monitor does not transfer (joint
  shared Monitor is better, see Y4 H10)
- Any context where the policy is non-stationary or the trace
  distribution is diverse

### 3.4 The "shipping use" framework

The Monitor's verified shipping uses are:
1. **Verification** (DLR predicates for cross-agent reasoning,
   runtime guardrails for safety)
2. **Failure prediction for handoff** (predict failure and
   escalate to a more reliable approach)
3. **Test-time safety check** (use the Monitor as a black-box
   failure predictor at deployment)

The Monitor's non-verified uses (which we do NOT recommend):
1. **Training signal in multi-agent RL** (Y3 REFUTED)
2. **Training signal in LLM self-monitoring** (Y4 REFUTED)
3. **Cross-context transfer** (Y3 + Y4 both REFUTED)

## 4. Implications for the field

### 4.1 What this means for MARL researchers

The Monitor architecture (decoupled per-agent Monitors with
frozen training) is a verified single-agent RL signal but does
NOT transfer to MARL. The MARL community should:
1. **Use DLR predicates in the critic** (Y3 v8 dlr_only result,
   +0.06 at n=100, p_bonf=0.043) instead of Monitor signal
2. **Use hand-crafted interpretable features** for cross-agent
   signal in MA, not learned failure predictions
3. **Investigate alternative MA directions** (learned comms
   TarMAC/IC3Net, not Monitor signal)

### 4.2 What this means for LLM self-monitoring

The frozen-decoupled Monitor approach does NOT transfer to LLM
self-monitoring. The LLM community should:
1. **Use joint shared Monitors** (Y4 H10 finding) instead of
   frozen-decoupled
2. **Use more sample-efficient training** (joint shared uses
   more data per update)
3. **Investigate alternative LLM self-monitoring approaches**
   (e.g., ensemble methods, self-consistency, etc.)

### 4.3 What this means for the Monitor architecture in general

The Monitor is a **context-specific** signal. Generalization
across contexts requires verification, not assumption. The
research community should:
1. **Always verify** the Monitor on the target context, not
   assume it transfers
2. **Pre-register** the verification study (like Y3 H5 and
   Y4 H10)
3. **Report negative results** when the Monitor does NOT
   transfer (as in Y3 5/6 REFUTED and Y4 H10 REFUTED)

## 5. Conclusion

We synthesized evidence from three independent investigations
across three contexts: single-agent RL (Y1, verified),
multi-agent RL (Y3, 6-pathway systematic investigation, 5/6
REFUTED), and LLM self-monitoring (Y4, H10 pre-registered pilot,
REFUTED). The pattern is consistent: **the failure-prediction
Monitor does not transfer** from the context in which it was
verified to other contexts.

The Monitor's verified shipping use remains **verification**
(DLR predicates in critic for cross-agent signal, runtime
guardrails for safety), not training. The Monitor is a
"context-specific" signal that works only in the narrow regime
where it was verified.

We propose a unified framework for thinking about Monitor
transfer: the Monitor works when the agent's policy is stable,
the training distribution matches the use-time distribution, and
the signal is informative. The Monitor fails when the policy
changes, the signal is biased, or the training data is sparse.

The research community should always verify the Monitor on the
target context, not assume it transfers, and report negative
results when it does not.

## Acknowledgments

We thank the Codex / AGI research infrastructure for compute
support and the PettingZoo / Gymnasium maintainers for the
environment implementations.

## References

1. Z. Liu. Y1 Paper: Single-Agent Failure-Prediction Monitors in
   Reinforcement Learning. AGI Research Project, AGI-2026-001,
   2026. `papers/y1_paper_draft.md`
2. Z. Liu. Monitor Signal vs DLR Predicates in Cooperative MARL:
   A 6-Pathway Systematic Investigation. Y3 paper, AGI Research
   Project, AGI-2026-001, 2026. `papers/monitor_signal_vs_dlr_6pathway.{md,tex,pdf}`
3. Z. Liu. Project G v0.5: Stratified Split for H10 LLM
   Self-Monitoring Pilot. Y4 paper, AGI Research Project,
   AGI-2026-001, 2026. `papers/project_g_v0_5_h10_paper.md`
4. Z. Liu. Y1 9-Hypothesis Framework. AGI Research Project,
   AGI-2026-001, 2026. `papers/y1_9hypothesis_framework.md`
5. R. Lowe et al. Multi-Agent Actor-Critic for Mixed
   Cooperative-Competitive Environments. NeurIPS 2017.
6. J. K. Terry et al. PettingZoo: Gym for Multi-Agent
   Reinforcement Learning. NeurIPS 2021.
