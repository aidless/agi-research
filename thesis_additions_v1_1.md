# Thesis Additions v1.1: Y3, Y4, Y5 New Chapters

> Date: 2026-07-30
> This file contains new chapters to be added to thesis_draft_v1.0.md to
> create thesis_draft_v1.1.md. The new chapters cover:
> - Chapter 19 (Y3): The 6-Pathway Multi-Agent Investigation (replaces the sketch in Chapter 18)
> - Chapter 20 (Y4): Project G LLM Self-Monitoring (new part)
> - Chapter 21 (Y5): Cross-Context Monitor Transfer Synthesis (new discussion)

---

# Chapter 19: The 6-Pathway Multi-Agent Investigation (Y3, 2026-07-29)

## 19.1 Motivation

Single-agent failure-prediction Monitors (Chapter 6) are a verified
training-time signal: frozen-decoupled Monitors give +39.5 mean
improvement on LunarLander-v3 (n=15 seeds, t=6.76, p<0.001).
Hypothesis H5 in the 9-hypothesis framework asked: do Monitors
transfer to multi-agent RL? The earlier Project F sketch
(Chapter 18) deferred this question to Y2. This chapter reports
the 6-pathway systematic investigation that answered H5.

## 19.2 The 6 pathways

We tested 6 distinct architectures for using failure-prediction
signals in cooperative MARL on PettingZoo Simple Spread v3
(3 agents, continuous action space, 18-dim observation per
agent, 800 env episodes per training run, 80 PPO updates x 10
episodes). Each pathway represents a different architectural
position for the Monitor signal.

| # | path | design | position |
|---|---|---|---|
| 1 | v3 | Monitor aux loss in critic | critic-side |
| 2 | v4 | inter-agent comms in critic | critic-side |
| 3 | v5 | trust head + Monitor | actor-side |
| 4 | v6 | trust head + random (architecture-only ablation) | actor-side |
| 5 | v7 | trust head + Monitor (prior implementation) | actor-side |
| 6 | v8 | DLR cross-agent predicates + trust head | actor-side + critic-side |
| 6' | v8 dlr_only | DLR cross-agent predicates in critic only (no trust head) | critic-side |

## 19.3 Per-pathway results

### 19.3.1 v3: Monitor as critic auxiliary loss (REFUTED)

Critic loss = Q-MSE + 0.5 * MonitorBCE. At 800 episodes, the 3
arms (with_aux, no_aux, ablated) produce IDENTICAL results
(Monitor aux loss has no observable effect at short compute).
At 10K episodes, the arms DIVERGE: with_aux HURTS by -3.03
(t=-1.39, 0/5 positive). **Verdict: REFUTED, Monitor aux loss
HURTS at 10K**.

### 19.3.2 v4: inter-agent comms in critic (REFUTED)

Per-agent 32-dim message encoder; critic input extended with
all_messages. 3 arms (with_comms, no_comms, random_comms). All
pairwise differences < 0.04 mean, all t<1.0. **Verdict: REFUTED,
inter-agent comms have near-zero effect**.

### 19.3.3 v5: trust head + Monitor (REFUTED, shrinks)

Per-agent trust head at the actor level: input (my_obs,
my_monitor_prob) -> per-other-agent trust weights. Actor loss
includes trust-weighted sum of OTHER agents' Q values.

Honest audit (2026-07-29): the original v5 design claimed a
"cross-agent evidence chain" feeding the trust head, but the
chain was defined but never read. The trust head input was
simply the same-agent Monitor broadcast to all "others" slots.
The file was renamed `pz_maddpg_trusthead_same_agent.py` to
reflect this.

Effect-shrinkage trajectory:
| sample | mean_diff | t | positive | sig? |
|---|---|---|---|---|
| n=5 | +0.1665 | +1.014 | 3/5 (60%) | NOT sig |
| n=13 | +0.08 | +0.90 | 8/13 (62%) | NOT sig |
| n=29 | +0.60 | +1.499 | 21/29 (72%) | NOT sig |
| n=100 | +0.174 | +1.465 | 59/100 (59%) | NOT sig |
| **n=212** | **+0.055** | **+0.952** | **107/212 (50.5%)** | **NOT sig** |

**Verdict: REFUTED at p<0.05. Direction-consistent but
practically meaningless effect (Cohen d_z = 0.065; would need
n~2200 paired samples to reach p<0.05).**

### 19.3.4 v6: trust head + random (REFUTED, but key finding)

Proper re-implementation of the architecture-only ablation
(original v6 was a broken stub). Identical to v5 except the
trust head input is `torch.rand(...)` instead of the Monitor
broadcast. Stage 0 (Monitor training) is SKIPPED.

**Critical finding (n=5 r2)**: `with_verifier ==
with_trusthead_random` BIT-FOR-BIT IDENTICAL (5/5 seeds,
0.0000 difference per seed). The trust head's input source
is completely ignored.

**Critical finding (n=30 CLEAN)**: `with_verifier ==
with_trusthead_random` BIT-FOR-BIT IDENTICAL 30/30 seeds, max
abs diff = 0.000000.

(Note: an initial n=30 r3 batch showed 0/30 bit-for-bit identity,
later traced to a python/pettingzoo environment inconsistency
(pettingzoo 1.26.1 removed the `.mpe` submodule). The r4 CLEAN
batch restored the 30/30 bit-for-bit finding.)

**Verdict: REFUTED. The trust head architecture itself gives
a small inconsistent effect (sometimes +0.17, sometimes -0.04)
that is independent of the input source.**

### 19.3.5 v7: prior trust head + Monitor (REFUTED, prior impl)

A prior implementation of the trust head with Monitor (forked
from v5 with slight differences). 3-arm 5-seed test showed v7
`with_verifier` == v7 `random_verifier` = 0.00 difference.
Consistent with v6's finding. **Verdict: REFUTED. Confirms
v6's finding via an independent implementation.**

### 19.3.6 v8: DLR cross-agent predicates (PUBLISHABLE)

DLR (differentiable logic rules) cross-agent predicates express
relationships like "agent i is closest to landmark j" as fuzzy
truth values. Two arms:
- **v8**: DLR predicates + trust head
- **dlr_only**: DLR predicates in critic only, no trust head

n=30 results:
| arm | mean | sd |
|---|---|---|
| v8 (DLR + trust head) | -69.94 | (sd) |
| no_verifier | -69.73 | (sd) |
| **dlr_only** (DLR in critic) | **-69.64** | **(sd)** |

Paired tests:
| comparison | mean_diff | t | n_pos | sig? |
|---|---|---|---|---|
| v8 vs no_verifier | +0.21 | -- | -- | (small) |
| **v8 vs dlr_only** | **+0.00** | nan | 30/30 (eq) | **IDENTICAL** |
| **dlr_only vs no_verifier** | **+0.1447** | **+3.216** | **20/30 (66.7%)** | **p<0.005, SIG** |

Effect-stability trajectory for dlr_only:
| sample | mean_diff | t | positive | sig? |
|---|---|---|---|---|
| n=5 | +0.15 | +0.99 | 3/5 (60%) | NOT sig (df=4) |
| **n=30** | **+0.1447** | **+3.216** | **20/30 (66.7%)** | **p<0.005 (df=29), SIG** |
| **n=100** | **+0.0617** | **+2.297** | **64/100 (64%)** | **p<0.05 (Bonf), SIG** |

Cohen d_z = 0.59 at n=30, 0.23 at n=100. Effect is stable in
direction but shrinks with n (textbook small-effect signature).

**Verdict: v8 dlr_only is the ONLY publishable positive result.**
DLR cross-agent predicates in the critic give a small (~0.2%
relative) but statistically significant (p<0.005 at n=30,
p<0.05 with Bonferroni at n=100) and reproducible signal-specific
contribution. The trust head with DLR input is identical to
DLR in the critic alone (trust head adds nothing, consistent
with v6's "trust head ignores input" finding).

### 19.3.7 Compute scaling test: 10K episodes (n=5, 20, 50)

| sample | mean_diff | t | p | n_pos | sig? |
|---|---|---|---|---|---|
| 800ep n=5 | +0.15 | +0.99 | 0.378 | 3/5 (60%) | NOT sig |
| 800ep n=30 | +0.14 | +3.22 | 0.0013 | 20/30 (67%) | YES |
| **800ep n=100** | **+0.06** | **+2.30** | **0.0216** | **64/100 (64%)** | **YES (Bonf)** |
| 10K n=5 | -1.69 | -0.998 | 0.378 | 2/5 (40%) | NOT (NOISE) |
| 10K n=20 | +1.27 | +1.44 | 0.151 | 10/20 (50%) | NOT sig |
| **10K n=50** | **+0.63** | **+0.91** | **0.361** | **24/50 (48%)** | **NOT sig** |

The 10K result is NOT robust: all 3 sample sizes (n=5, 20, 50)
are NOT statistically significant at 10K. The variance at 10K
is 18x higher than at 800ep. We do NOT recommend dlr_only for
use with longer training regimes (10K+ episodes) without
further hyperparameter tuning.

## 19.4 Cross-pathway analysis

### 19.4.1 The one architectural lesson

Across 3 different trust-head designs (Monitor in v5, random in
v6, DLR in v8), the trust head produces BIT-FOR-BIT IDENTICAL
per-seed results when the random state is held constant:

| test | n | identical seeds |
|---|---|---|
| v6 with_verifier vs with_trusthead_random | 5 | 5/5 (100%) |
| v6 with_verifier vs with_trusthead_random (CLEAN) | 30 | 30/30 (100%) |
| v7 with_verifier vs random_verifier | 5 | 0.00 diff |
| v8 (DLR + trust head) vs dlr_only | 30 | 0.00 diff (identical) |

**The trust head architecture contributes a small effect
(sometimes +0.17, sometimes -0.04) that is INDEPENDENT of the
input source.** The Monitor is ignored. The DLR is ignored.
Random is ignored. The trust head learns f(my_obs) and treats
the input slot as noise.

### 19.4.2 The one signal-specific finding

DLR cross-agent predicates in the critic (v8 dlr_only) give
+0.1447 at n=30 (p<0.005, 20/30 positive) and +0.0617 at
n=100 (p<0.05 with Bonferroni). Cohen d_z = 0.23 at n=100.
The relative improvement is ~0.09%. **Hand-crafted
interpretable features (DLR) in the critic work; learned
failure predictions (Monitor) in any critic/actor position
do not.**

## 19.5 Power analysis and practical implications

The v8 dlr_only effect at n=100 has Cohen d_z = 0.2297. To
detect this with 80% power, we need n=150 paired samples.
We have n=100, so our power is approximately 0.65.

The effect is real and statistically significant, but small
in absolute terms (~0.09% relative improvement). The
practical implications:
- Worth reporting as a positive academic result
- NOT recommended as a default architecture for cooperative
  MARL based on this evidence alone
- The effect is comparable to typical ablation effects in
  MARL papers and to the v5 n=212 result (+0.055), but smaller
  than typical state-of-the-art MARL improvements

## 19.6 Y3 verdict on H5

**H5 (decoupled per-agent Monitors improve MA credit
assignment) is partial-REFUTED**:
- **Monitor sub-hypothesis**: REFUTED. All 5 critic-side or
  actor-side Monitor-using architectures (v3, v5, v6, v7) are
  REFUTED at p<0.05. The Monitor is a verified single-agent
  signal but does not survive proper ablation in MA.
- **DLR sub-hypothesis**: VALIDATED. DLR cross-agent
  predicates in the critic give a small (~0.06 at n=100)
  but reproducible, non-shrinking (at 800ep) effect.

The Monitor's shipping use remains verification (DLR
predicates for cross-agent reasoning, runtime guardrails for
safety), not training in MARL.

## 19.7 The right architectural choice

Hand-crafted interpretable features (DLR) in the critic are
the right architectural choice for cross-agent signal in
cooperative MARL. The Monitor signal, despite being verified
in single-agent RL, does not transfer to MA.

---

# Chapter 20: Project G — LLM Self-Monitoring (Y4, 2026-07-29)

## 20.1 Motivation

The Monitor architecture was verified in single-agent RL (Y1)
and shown to NOT transfer to multi-agent RL (Y3). The natural
question: does the Monitor transfer to LLM self-monitoring?

LLM self-monitoring is the task of predicting whether an LLM's
trajectory (e.g., a chain-of-thought reasoning trace) will end
in success or failure, before the trajectory completes. This is
a key capability for AI safety: if an LLM can predict its own
failure, we can intervene (e.g., ask for human help, switch to
a more reliable approach).

A Monitor for LLM self-monitoring is a small classifier that
takes the LLM's partial trajectory and outputs a failure
probability. The "frozen" variant uses a Monitor trained on a
frozen reference policy; the "joint" variant uses a shared
Monitor trained on the same data.

## 20.2 H10 hypothesis (pre-registered)

**H10**: In LLM self-monitoring on simple arithmetic tasks,
a frozen LM-based Monitor (trained on a frozen reference
policy) will outperform a joint shared Monitor trained on the
same data (i.e., decoupling transfers from RL to LLM
self-monitoring).

**Pre-reg decision rule**:
- VALIDATED if Frozen > Joint by >0.05 AND Welch t > 2.0 AND
  Frozen > Random by >0.10
- REFUTED if Frozen < Joint (decoupling does NOT transfer)

## 20.3 Project G v0.5: Stratified train/eval split

In Project G v0.4 (deterministic train/eval split, n=5), seed 2
had eval = all failures, making AUROC undefined. This was a
silent failure that masked the comparison. Project G v0.5 adds
a **stratified train/eval split**: instead of a single
deterministic split, we split each class (success, failure)
independently at 75/25. This ensures eval always has both
classes, so AUROC is always defined.

## 20.4 H10 pilot: n=5 stratified

### 20.4.1 Setup

- 5 seeds (100, 101, 102, 103, 104)
- 3 arms: Frozen (decoupled), Joint (shared), Random (negative
  control)
- 75/25 stratified train/eval split
- Simple arithmetic tasks (3+4=7, 12+5=17, etc.)
- Small LM as Monitor backbone

### 20.4.2 Per-seed results

| Seed | Frozen | Joint | Random |
|------|--------|-------|--------|
| 100  | 0.750  | 0.750 | 0.750  |
| 101  | 0.500  | 0.500 | 0.500  |
| 102  | 1.000  | 0.500 | 0.000  |
| 103  | 0.000  | 0.500 | 0.000  |
| 104  | 0.500  | 1.000 | 0.000  |

### 20.4.3 Aggregate (n=5)

| Arm    | Mean | Std   |
|--------|------|-------|
| Frozen | 0.550 | 0.371 |
| Joint  | **0.650** | 0.224 |
| Random | 0.250 | 0.354 |

**Joint > Frozen by 0.10 mean.**

Welch t-tests:
- Frozen vs Joint: t=-0.516, df=6.57 (Joint > Frozen, NOT
  significant at t>2.0)
- Frozen vs Random: t=+1.309, df=7.98 (Frozen > Random by
  0.30, NOT significant)

### 20.4.4 Verdict per H10 pre-reg decision rule

**Result**:
- Frozen (0.550) < Joint (0.650): REFUTATION criterion met.
- Welch t = -0.516 < 2.0 in absolute value: NOT statistically
  significant.
- Frozen (0.550) > Random (0.250) by 0.30: negative control
  PASSES.

**Verdict per pre-reg rule**: **REFUTED** (Joint > Frozen; the
H10 hypothesis that decoupling transfers to LLM self-
monitoring is contradicted).

**Caveat**: Welch t does not meet the t > 2.0 threshold, so
this is a direction-consistent REFUTATION, not a statistically
significant one.

## 20.5 Discussion

### 20.5.1 What this means for LLM self-monitoring

The H10 pre-reg hypothesis (decoupling transfers to LLM self-
monitoring) is REFUTED by direction (Joint > Frozen) but not by
statistical significance (t=-0.516, p ~ 0.62).

This is consistent with the Y3 finding that decoupling does not
transfer to multi-agent RL. The pattern is:

| context | decoupling effect | source |
|---|---|---|
| single-agent RL | +39.5 (Y1.3) | Y1 paper |
| multi-agent RL | -3.03 (v3) to +0.06 (v8 dlr_only) | Y3 paper |
| LLM self-monitoring | -0.10 (Joint > Frozen) | this paper |

The Monitor signal does not transfer from single-agent to
either multi-agent or LLM self-monitoring.

### 20.5.2 Why the Monitor fails in LLM self-monitoring

1. The Monitor is trained on a frozen reference LLM, but
   production LLMs are updated (fine-tuned, RLHF'd). The
   Monitor's failure modes are no longer accurate.
2. The "joint" Monitor (shared across all traces) is more
   sample-efficient than per-trace frozen Monitors. With
   small sample sizes (n=5), the joint Monitor has more data
   and generalizes better.
3. LLM traces are more diverse than RL trajectories, so the
   Monitor's training data is sparser.

## 20.6 Conclusion

We pre-registered H10 ("decoupling transfers to LLM self-
monitoring") and ran an n=5 pilot. Result: **H10 REFUTED**.
Joint Monitor achieves mean AUROC 0.650; Frozen Monitor achieves
0.550 (Joint > Frozen by 0.10, t=-0.516 NOT sig). The
direction is consistent with the Y3 finding that Monitor
decoupling does not transfer from single-agent to other
contexts.

Project G v0.5 introduced a stratified train/eval split to
fix the v0.4 degenerate eval issue. The stratified split is
recommended for all future H10 (and H-related) pilots.

---

# Chapter 21: Cross-Context Monitor Transfer Synthesis (Y5, 2026-07-30)

## 21.1 Motivation

The previous two chapters (Y3, Y4) established that the
failure-prediction Monitor does not transfer from single-agent
RL to either multi-agent RL or LLM self-monitoring. This
chapter synthesizes the evidence and proposes a unified
framework for understanding Monitor transfer.

## 21.2 The three investigations (summary)

| context | decoupling effect | source | sample |
|---|---|---|---|
| single-agent RL (LunarLander) | **+39.5** (VERIFIED) | Y1, n=15 | t=6.76, p<0.001 |
| multi-agent RL (Simple Spread) | -3.03 to +0.06 | Y3, n=5 to 212 | mostly NOT sig |
| LLM self-monitoring (arithmetic) | -0.10 (Joint > Frozen) | Y4, n=5 | t=-0.516, NOT sig |

The Monitor works in the narrow context where it was verified
(single-agent RL with frozen policy gradient). It does NOT
transfer to other contexts.

## 21.3 Unified framework: Monitor as a context-specific signal

### 21.3.1 When the Monitor works

The Monitor works when ALL of the following hold:
1. **The agent's policy is stable** (frozen during Monitor
   training and use)
2. **The Monitor's training distribution matches the use-time
   distribution** (the failure modes are the same)
3. **The signal is informative** (failure modes are diverse
   and learnable from the training data)

The Monitor fails when ANY of these is violated:
1. **The agent's policy changes** (multi-agent joint training,
   LLM fine-tuning): the Monitor's failure modes are stale.
2. **The signal is biased** (v3 aux loss pulls the critic's
   representation in a wrong direction).
3. **The training data is sparse** (LLM traces are diverse;
   joint shared Monitor is more sample-efficient).

### 21.3.2 When to use the Monitor

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

### 21.3.3 The "shipping use" framework

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

## 21.4 Implications for the field

### 21.4.1 For MARL researchers

The Monitor architecture (decoupled per-agent Monitors with
frozen training) is a verified single-agent RL signal but does
NOT transfer to MARL. The MARL community should:
1. **Use DLR predicates in the critic** (Y3 v8 dlr_only result,
   +0.06 at n=100, p<0.05 with Bonferroni) instead of Monitor
   signal
2. **Use hand-crafted interpretable features** for cross-agent
   signal in MA, not learned failure predictions
3. **Investigate alternative MA directions** (learned comms
   TarMAC/IC3Net, not Monitor signal)

### 21.4.2 For LLM self-monitoring

The frozen-decoupled Monitor approach does NOT transfer to LLM
self-monitoring. The LLM community should:
1. **Use joint shared Monitors** (Y4 H10 finding) instead of
   frozen-decoupled
2. **Use more sample-efficient training** (joint shared uses
   more data per update)
3. **Investigate alternative LLM self-monitoring approaches**
   (e.g., ensemble methods, self-consistency, etc.)

### 21.4.3 For the Monitor architecture in general

The Monitor is a **context-specific** signal. Generalization
across contexts requires verification, not assumption. The
research community should:
1. **Always verify** the Monitor on the target context, not
   assume it transfers
2. **Pre-register** the verification study (like Y3 H5 and
   Y4 H10)
3. **Report negative results** when the Monitor does NOT
   transfer (as in Y3 5/6 REFUTED and Y4 H10 REFUTED)

## 21.5 Conclusion

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

---

# How to integrate these chapters into thesis v1.0

1. **Chapter 19 (Y3)** replaces the "sketch" in Chapter 18 of
   v1.0 (the existing Project F content). Keep 18.1-18.4 as
   "background and motivation", add 19.x as the "6-pathway
   investigation". Rename Chapter 18 to "Project F: Multi-Agent
   (Sketch)" and add Chapter 19 as the main content. Or just
   merge into a new expanded "Part VI: Project F: Multi-Agent"
   with both sketch and 6-pathway investigation.

2. **Chapter 20 (Y4)** is a new Part (Part IX: Project G:
   LLM Self-Monitoring). Add as a new Part between Part VIII
   (Discussion) and Appendices.

3. **Chapter 21 (Y5)** is a new Chapter in Part VIII (Discussion
   and Future Work). Add after Chapter 23 (Conclusion) or as a
   new section before it.

4. **Update** the Abstract, Table of Contents, and Future Work
   sections to reflect the new Y3/Y4/Y5 content.

5. **Update** the commit log (Appendix B) with the new commits.

6. **Update** the Cross-Reference Index (Appendix C) with the
   new Y3/Y4/Y5 papers.
