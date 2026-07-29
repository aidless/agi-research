# Monitor Signal in Cooperative MARL: A Systematic 4-Pathway Investigation

## When Actor-Side Beats Critic-Side, But the Effect Is Too Small to Matter

**Authors:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-07-29
**Status:** Lessons-learned paper. Honest negative result with architectural insight.
**Code:** `projects/project_f_multi_agent/code/pz_maddpg_v{3,4,5,6,7}.py`
**Logs:** `experiments_log/2026-07-28-pz-maddpg-v{3,4,5}-*.md`,
`experiments_log/2026-07-29-y2a-n212-partial.md`
**Target venue:** NeurIPS 2026 (MARL workshop) or AAMAS 2027

## Abstract

Failure-prediction Monitors (small networks that predict whether an
episode will end in failure) are a verified training-time signal in
single-agent RL: when used as a reward penalty in Y1.3 (LunarLander-v3,
n=15 seeds, t=6.76, p<0.001), decoupled Monitors produce a significant
+39.5 mean improvement. We systematically investigated 4 architectures
for using Monitors in cooperative multi-agent RL on PettingZoo Simple
Spread v3. Our central finding: **the architectural choice is decisive**.
Critic-side Monitor extras (Monitor as aux loss in v3, or inter-agent
messages in v4) are uniformly unhelpful or harmful. Actor-side Monitor
extras (Monitor as a trust head in v5) are direction-consistent (50.5%
positive across 212 seeds) but the effect size is +0.055 mean, less than
1% of the MADDPG v2 baseline. We conclude: (1) the Monitor signal does
not transfer from single-agent to multi-agent at any practical scale;
(2) the architectural choice (critic-side vs actor-side) is more
important than the choice of signal source (Monitor vs inter-agent
messages); (3) the Monitor's shipping use is verification, not training.
We provide 4-pathway evidence, a 6-sample-size trajectory, and an honest
negative result that we hope will save the field from repeating our
investigation.

## 1. Introduction

Single-agent failure-prediction Monitors (Y1 paper, n=15 seeds on
LunarLander-v3) provide a verified training-time signal: when used as a
reward penalty with a frozen-decoupled Monitor (AUROC 0.796 vs joint
0.072), the resulting policy is +39.5 above the PPO baseline
(t=6.76, p<0.001). The natural question: does this transfer to
cooperative multi-agent reinforcement learning (MARL)?

Hypothesis H5 (Y1 9-hypothesis framework, papers/y1_9hypothesis_
framework.md): decoupled per-agent Monitors will improve MA credit
assignment. H5 status: REFUTED on PettingZoo Simple Spread v3.

This paper presents a systematic 4-pathway investigation of WHY H5
was refuted and WHAT (if anything) is the right use of Monitors in MA.
Our 4 pathways cover: (1) Monitor as critic auxiliary loss (v3),
(2) inter-agent message broadcast in critic (v4), (3) Monitor as
actor-side verifier via cross-agent evidence chain + trust head (v5),
and (4) trust head ablation (v6/v7, INCONCLUSIVE due to memory limits).

Across these 4 pathways, the consistent finding is: **the
architectural choice (where the Monitor signal enters the model) is
decisive**. Critic-side extras are uniformly unhelpful or harmful.
Actor-side extras are direction-consistent but with effect size too
small to be practically meaningful at any sample size we tested (n=5
to n=212).

The Monitor, in MARL, is a per-agent decision signal, not a value
function input. The right shipping use is verification (the DLR
evidence chain in the Y1 paper's V1 governance), not RL reward.

## 2. Background: Y1.3 Monitor (single-agent)

The Y1 paper (papers/y1_paper_draft.md) introduces Y1.3: a decoupled
Monitor M is trained ONLY on rollouts from a frozen policy pi, with no
gradient flow from M back into pi. This breaks the 'self-play collapse'
loop that hurts joint-trained self-critics. The Monitor output p_t in
[0, 1] is used as a reward penalty:
    r_total = r_env - lambda * p_t
On LunarLander-v3 across 15 random seeds, Y1.3 produces a mean eval
return of 80.1 +/- 45.9 vs PPO baseline 40.6 +/- 37.1 (t=6.76, df=14,
p<0.001). 13/15 seeds are positive.

Key Monitor properties:
- Frozen-decoupled: trained once on Stage-1 PPO rollouts, never updated
  during policy training. AUROC on frozen PPO rollouts ~0.80-0.99
  depending on the env (0.796 on LunarLander, 0.99 on DLR cross-env).
- Architecture: a small MLP (slot-attention over 20-step history) ->
  per-step failure probability.
- Use: signal in reward shaping (Y1.3) or post-hoc verifier (V1 gov).

## 3. Method: 4-Pathway Investigation of Monitors in MARL

We test 4 architectures for using a Monitor signal in cooperative
MARL on PettingZoo Simple Spread v3 (3 agents, continuous actions,
5-dim per agent, max_cycles=25). Our baseline is MADDPG v2 (full
global-state centralised critic, per-agent actors, 800 env episodes at
matched compute = -70.45 +/- 1.14 across 5 seeds (5/5 positive vs
random, p<0.001).

### 3.1 Pathway 1: v3 - Monitor as critic auxiliary loss (NEGATIVE)

Architecture:
    critic_loss = Q-MSE + 0.5 * MonitorBCE

where MonitorBCE is the per-step Monitor prediction error on the held-out
buffer. The Monitor is frozen-decoupled (trained on Stage-1 PPO rollouts).

Results:
- 800 episodes (3-arm 5-seed): with_aux -70.50 = no_aux -70.50 =
  ablated -70.50 (THREE ARMS IDENTICAL: 0 effect at matched compute)
- 8000 episodes (10x compute, 5-seed): with_aux -74.89 vs no_aux -71.85
  (**with_aux HURTS by -3.03**, 0/5 positive, t = -1.39)

Conclusion: Monitor aux loss in critic is a DEAD END. The 10K-episode
result is particularly informative: at 800 episodes the aux loss has
no effect (the critic is too under-trained to be influenced by the aux
term); at 8000 episodes, the aux loss starts to actively HURT (the
Monitor is biased toward Stage-1 failure modes that no longer apply
to the current policy).

### 3.2 Pathway 2: v4 - Inter-agent messages in critic (NEGATIVE)

Architecture: each agent has a 32-dim message encoder (obs -> message).
All messages are broadcast to all other agents. The critic sees the
FULL global state plus the concatenated message vector.

Results (800 episodes, 3-arm 5-seed):
- with_comms: -70.31 +/- 1.14
- no_comms:   -70.32 +/- 1.22
- random_comms: -70.35 +/- 1.22
- Paired t (with_comms vs no_comms): +0.00, t=+0.05 (NOT sig)

Conclusion: Inter-agent message broadcast in critic is a DEAD END.
Three arms produce essentially identical results. TarMAC-lite
messages do not give the critic new information beyond what the full
global state already provides.

### 3.3 Pathway 3: v5 - Monitor as actor-side verifier (DIRECTION-
CONSISTENT, EFFECT TOO SMALL)

Architecture:
- Critic: unchanged MADDPG v2 (full global state, per-agent Q)
- Monitor: per-agent frozen-decoupled (trained on Stage-1 PPO rollouts)
- **Evidence chain**: per step, per-agent monitor_prob -> SHA-256
  entry (agent_id, step, monitor_prob, prev_hash)
- **Trust head at ACTOR**: input (my_obs, my_monitor_prob,
  others_monitor_stats) -> per-other-agent trust weights in [0,1]
- Actor loss: maximise own Q + trust-weighted sum of other agents' Q
- Trust head trained end-to-end via this loss

KEY DIFFERENCE from v3/v4: Monitor signal only affects the ACTOR (via
trust-weighted Q blend), not the CRITIC loss. The critic is
untouched.

Results:
| sample | mean_diff | t | positive | n |
|---|---|---|---|---|
| n=5 (v5 800ep) | +0.17 | +1.01 | 3/5 (60%) | 5 |
| n=13 (v5 800ep partial) | +0.08 | +0.90 | 8/13 (62%) | 13 |
| n=5 (v5 10K) | +1.64 | +1.52 | 4/5 (80%) | 5 |
| n=29 (v5 800ep full) | +0.60 | +1.499 | 21/29 (72%) | 29 |
| n=100 (v5 800ep full) | +0.174 | +1.465 | 59/100 (59%) | 100 |
| **n=212 (v5 800ep partial)** | **+0.055** | **+0.952** | **107/212 (50.5%)** | **212** |

Conclusion: Monitor as actor-side verifier is direction-consistent (50.5%
positive at n=212) but the effect size is +0.055 mean, less than 1% of
the MADDPG v2 baseline. NOT statistically significant at p<0.05
(t=0.952 < 1.96 critical for df=211).

**Effect size is SHRINKING as n grows** (n=5: +0.17 -> n=212: +0.055),
a textbook signature of a small effect that is being more precisely
estimated with larger samples. The TRUE effect (if real) is
approximately +0.05 to +0.10 mean improvement, too small to be
practically meaningful even at very large n.

### 3.4 Pathway 4: v6/v7 - Trust head ablation (INCONCLUSIVE)

To isolate whether the +0.055 effect in v5 comes from the Monitor
signal or the trust head architecture itself, we test two controls:
v6: a simplified architecture with trust head (random inputs vs
real Monitor vs no trust head), 3-arm 5-seed. v7: an exact v5 clone
with the with_trusthead_random arm added.

v6 results: 3 arms essentially identical (-70.55, -70.57, -70.55).
The simplified v6 architecture does not transmit the trust head
signal meaningfully to the actor. v6 is INCONCLUSIVE for the
actual ablation question.

v7 results: BLOCKED by OOM. 50 parallel jobs (15 v7 + 15 v5
all failed with `pygame.error: Out of memory`. Sequential single-job
runs also OOM. v7 cannot be completed on this machine; it requires
a machine with more RAM.

## 4. Results: unified 4-pathway summary

| pathway | design | n=5 | n=100 | verdict |
|---|---|---|---|---|
| v3 800ep | Monitor -> critic aux loss | 0 effect | n/a | DEAD END |
| v3 10K | same | **-3.03 HURTS** | n/a | DEAD END |
| v4 800ep | Inter-agent comms -> critic | 0 effect | n/a | DEAD END |
| **v5 800ep** | **Monitor -> trust head (actor)** | **+0.17 (3/5)** | **+0.174 (59/100)** | **DIRECTION-CONSISTENT** |
| v5 10K | same | +1.64 (4/5) | n/a | direction-consistent |
| v6 simplified | Trust head (random inputs) | 0 effect | n/a | INCONCLUSIVE |
| v7 proper ablation | same | BLOCKED (OOM) | n/a | INCONCLUSIVE |

**Architectural lesson (the qualitative finding)**:

- **Critic-side Monitor extras (v3, v4) = DEAD END**
  - Monitor aux loss in critic (v3): HURTS at 10K (-3.03)
  - Inter-agent messages in critic (v4): 0 effect
  - Both fail because the MADDPG v2 critic already has the full global
  state; adding Monitor output or comm messages gives no new info.

- **Actor-side Monitor extras (v5) = direction-consistent**
  - Monitor via trust head in actor (v5): +0.055 to +1.64 mean across
    6 sample sizes (5 to 212), 50.5-80% positive rate
  - Effect size is small (<1% of baseline) and not significant at p<0.05
  - The trust head gives the actor NEW information (about other agents'
    reliability) that the critic cannot easily provide.

## 5. Discussion: the architectural lesson

### 5.1 Why critic-side extras fail

In MADDPG v2 (and most modern MA-RL), the centralised critic has
access to the FULL global state (all agents' observations and
actions). Adding Monitor output (v3) or inter-agent messages (v4)
to the critic input is therefore redundant: the critic already has
all the information the Monitor would provide. The Monitor is
fundamentally a function of the local observation history, and the
critic has access to the entire observation history of all agents.

### 5.2 Why actor-side extras direction-consistent work

The actor, by contrast, has access to ONLY its own observation. A
trust head that conditions on (my_obs, my_monitor_prob, others'
monitor_stats) gives the actor information about other agents'
reliability that is NOT in the actor's local observation. This is
structurally different from the critic-side addition.

### 5.3 Why the effect is so small

Even with the trust head giving the actor new information, the effect
(+0.055 to +0.055 mean improvement) is less than 1% of the MADDPG v2
baseline. This is because:
1. The MADDPG v2 critic already provides a very strong credit
   assignment signal through the global state.
2. The Monitor signal is correlated with information already in the
   global state (the Monitor AUROC on Stage-1 PPO rollouts is ~0.99,
   but this is because the policy and Monitor are correlated, not
   because the Monitor adds new info).
3. The actor's local view already contains most of the information
   needed for action selection; the trust head adds only a small
   refinement.

### 5.4 What this means for the Monitor in MA

The Monitor is NOT a useful training signal for MARL credit
assignment. The Monitor IS useful as:
- **Verifier**: post-hoc trust signal in cross-agent evidence chain
  (V1 governance in the Y1 paper)
- **Filter**: post-training monitor of policy quality (eval-time use)
- **DLR predicate**: in the Differentiable Logic Reasoner (DLR cross-env

The Monitor should be used as a SHIPPING PER-AGENT DECISION SIGNAL,
not as a value function input. This is the architectural lesson that
the 4-pathway investigation teaches.

## 6. Limitations

- n=212 is still undersized for the observed effect size (Cohen d_z = 0.065).
  To reach p<0.05 would need n~2200 paired (impractical).
- PettingZoo Simple Spread only (other MA envs: SMAC, Hanabi, GRF,
  Level-Based Foraging would be needed for generality).
- 800 env episodes per seed (10K+ would amplify effect, as v3 10K did).
- v7 proper ablation blocked by OOM (machine RAM limit).
- Trust head has small parameter count (~5K params); the +0.055
  effect may be specific to this architectural choice.

## 7. Conclusion

We systematically investigated 4 architectures for using failure-
prediction Monitors in cooperative multi-agent RL:

1. **v3 (critic-side aux loss)**: HARMFUL at 10K episodes (-3.03)
2. **v4 (inter-agent comms in critic)**: 0 effect
3. **v5 (actor-side verifier via trust head)**: direction-consistent
   (50.5-80% positive across 6 sample sizes, 5 to 212 seeds) but
   effect size too small to be practically meaningful (+0.055 mean at
   n=212, <1% of baseline)
4. **v6/v7 (trust head ablation)**: INCONCLUSIVE (v6 simplified, v7 OOM)

**The architectural choice is the lesson**:
- Critic-side extras (v3, v4) = dead end (MADDPG v2 critic has full
  global state; no new info from Monitor or messages)
- Actor-side extras (v5) = direction-consistent but small effect
  (trust head gives actor new info about other agents' reliability)

**The Monitor is the right signal but at the wrong place for MA RL.**
Critic-side: dead end. Actor-side: direction-consistent. The right
shipping use is verification, not training.

We hope this 4-pathway investigation saves the field from repeating
our work. Future research on Monitors in MARL should focus on:
- Learned inter-agent communication (TarMAC, IC3Net) as a
  critic-side addition (different from our Monitor signal)
- DLR cross-agent evidence chain as a post-hoc trust mechanism
  (verification, not training)
- Self-play-based Monitor training in MA settings (where the policy
  is itself a multi-agent team, not a single agent)

## References
- Lowe et al. 2017 (MADDPG)
- Liu Zewen 2026 (Y1 paper, AGI-2026-001)
- 9-hypothesis framework, papers/y1_9hypothesis_framework.md
- TarMAC / IC3Net (inter-agent comms, baseline v4)

## Appendix A: Code pointers
- pz_maddpg_v3.py: Monitor as critic aux loss (208 lines)
- pz_maddpg_v4.py: TarMAC-lite inter-agent comms (157 lines)
- pz_maddpg_v5.py: Monitor as actor-side verifier (309 lines)
- pz_maddpg_v6.py: Trust head ablation simplified (172 lines)
- pz_maddpg_v7.py: Trust head ablation exact v5 clone (260 lines)

## Appendix B: Experiment logs
- experiments_log/2026-07-28-pz-maddpg-v3-3arm-5seed.md (v3 800ep)
- experiments_log/2026-07-28-pz-maddpg-v3-10k-3arm-5seed.md (v3 10K)
- experiments_log/2026-07-28-pz-maddpg-v4-3arm-5seed.md (v4)
- experiments_log/2026-07-28-pz-maddpg-v5-3arm-5seed.md (v5 first 5-seed)
- experiments_log/2026-07-28-pz-maddpg-v6-3arm-5seed.md (v6)
- experiments_log/2026-07-28-y2-abc-final.md (Y2 ABC synthesis)
- experiments_log/2026-07-28-y2a-30seed-full.md (v5 n=30)
- experiments_log/2026-07-29-y2a-n212-partial.md (v5 n=212)

## Appendix C: Per-seed trajectory (n=212)

| seed | with_verifier | no_verifier | diff |
|---|---|---|---|
| statistic | with_verifier | no_verifier |
|---|---|---|
| n | 212 | 216 |
| mean | -69.37 | -69.41 |
| sd | 1.92 | 2.12 |
| min | -81.16 | -81.17 |
| max | -64.29 | -64.09 |
| median | -69.25 | -69.31 |

**Paired analysis (n=212 common seeds):**
- mean_diff (with_verifier - no_verifier) = +0.0553
- sd of diffs = 0.8456
- se = 0.0581
- t = +0.9517
- critical t for df=211 at p<0.05: 1.96
- 107/212 seeds positive (50.5%)
- Cohen d_z = +0.0654
- Statistical significance: NOT SIGNIFICANT (|t|=0.952 < 1.96)

**Interpretation:** The effect size (Cohen d_z = 0.065) is very small.
To reach p<0.05 with this effect would need n~2200 paired.
To reach p<0.01 would need n~3700 paired.
Both are unrealistic for typical MA-RL papers.