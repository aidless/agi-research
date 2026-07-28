# Phase 2 Paper Outline — Decentralized Monitors for Multi-Agent Coordination

> Status: Outline draft (2026-07-28)
> Target venue: AAMAS 2028 or NeurIPS 2028
> Honest framing throughout (per user feedback)
> Estimated pages: 8-10 main + 4 appendix

## Honest framing preface

This outline describes **future work**. Unlike Y1 paper (which has
empirical results), Phase 2 has **no implementation yet**. Every claim
in this outline is a **proposal**, not a result. We use this outline to:

1. Identify research questions before implementation
2. Define falsifiable hypotheses for Y2 experiments
3. Document what we honestly do not know
4. Plan a 12-month Y2 timeline

Treat all "we will" statements as proposals pending Y2 implementation.

---

## §1 Introduction

### 1.1 The single-agent ceiling

Y1 paper validates Y1.3 (decoupled Monitor as training-time regularizer)
on LunarLander-v3 with 15 seeds, p<0.001. This is a single-agent result.

A natural next question: **does the decoupling insight extend to multi-agent
settings?** In particular:

- If each agent has its own Monitor, do they generalize across agents?
- Can shared world models coordinate Monitors without a central server?
- Does the decoupling failure mode (joint-trained Monitor) reappear in
  multi-agent credit assignment?

**Honest framing**: We have not yet run any multi-agent experiment. The
research questions below are forward-looking.

### 1.2 What multi-agent self-monitoring could look like

In a cooperative multi-agent setting (e.g., particle env navigation,
Hanabi, StarCraft multi-agent), each agent `i` has:

```
State: s_i (own observation)
Action: a_i
History: h_i = (s_0, a_0, ..., s_t)
Monitor_i: M_i(h_i) -> P(failure of agent i)
World model: W_i predicts next state given (s_i, a_i, joint_action)
```

Decentralized coordination means each agent uses its own Monitor and
world model, with communication limited to a shared symbolic layer
(e.g., via DLR predicates).

### 1.3 What we expect to find (predictions)

Hypothesis **H2**: In a cooperative multi-agent setting, **decentralized
decoupled Monitors trained on each agent's frozen policy outperform
jointly-trained shared Monitors**.

We expect this because the Y1 result (single-agent H1) shows decoupling
preserves discrimination. If the H1 mechanism generalizes, H2 should
hold.

**Honest note**: H2 is a hypothesis. It may fail if the multi-agent
credit assignment problem is fundamentally different from single-agent
distributional shift. We plan to test and report either outcome.

### 1.4 What this paper would contribute

If H2 holds:
- Decentralized Monitor coordination protocol
- Cross-agent knowledge transfer via shared symbolic layer
- A 2-agent or N-agent benchmark demonstrating the protocol

If H2 fails:
- A clear negative result: decoupling helps single-agent but not multi-agent
- Analysis of why credit assignment breaks the decoupling mechanism

Either outcome is publishable.

---

## §2 Background and Related Work

### 2.1 Multi-agent reinforcement learning (MARL)
- QMIX (Rashid 2018), MADDPG (Lowe 2017), COMA (Foerster 2018)
- Parameter sharing vs decentralized execution
- Credit assignment in cooperative settings

### 2.2 Multi-agent self-monitoring
- Almost no literature on decoupled Monitors in multi-agent
- Closest: "Self-Consistency" (Wang 2022) but for LLM reasoning, not RL
- Gap: this is a frontier

### 2.3 The Archimedes Project's single-agent foundation
- H1 ablation (5 seeds, frozen > joint Monitor)
- Y1.3 (15 seeds, training-time regularizer, p<0.001)
- DLR cross-env (4 envs, 97.8% mean accuracy)

### 2.4 Communication in MARL
- TarMAC (Das 2019), IC3Net (Singh 2019), NDQ (Wang 2020)
- Communication via learned messages vs symbolic predicates

---

## §3 Problem formulation

### 3.1 Decentralized POMDP

Each agent `i` has:
- Observation `o_i` from environment
- Action `a_i` from policy `pi_i`
- Local observation history `h_i`
- Communication message `m_i` to shared channel

The joint state is `s = (s_1, ..., s_N)`. Each agent's objective is to
maximize a shared team reward `R(s, a_1, ..., a_N)`.

### 3.2 Failure definition

A **joint failure** is a state where the team reward is below a threshold
(e.g., `R < 0` for navigation tasks).

Each agent's Monitor predicts its **own failure contribution** to the
joint failure: `P_i = P(this agent's action contributed to joint failure)`.

### 3.3 The decoupling question

For each agent `i`, should Monitor `M_i` be:
- **Decoupled**: trained on agent `i`'s frozen policy
- **Joint**: trained alongside the current policy
- **Shared**: shared across all agents (parameter sharing)

Y1 result predicts decoupled > joint for single-agent. Does it hold for
multi-agent?

---

## §4 Architecture proposal

### 4.1 Decentralized Monitor coordination (DMC)

```
For each agent i:
  M_i: local Monitor trained on frozen local policy
  W_i: local slot world model

Shared channel:
  C: DLR predicates broadcast from each agent (e.g., "agent_1 safe", "agent_2 in danger")
  Joint failure predictor F(C) -> P(joint failure)

Each agent's training:
  shaped_reward_i = env_reward + lambda * (1 - M_i(h_i)) * F(C)
```

The Monitor's failure probability *combined with* the joint failure
prediction shapes each agent's reward.

**Honest note**: This architecture is **untested**. The DLR broadcast
pattern requires 2+ agents to coordinate predicates; current DLR is
single-agent.

### 4.2 Cross-agent symbolic knowledge transfer

If agents learn complementary predicates (e.g., agent 1 knows about
"near_goal", agent 2 knows about "obstacle_ahead"), sharing predicates
should improve joint performance.

This is testable in particle env navigation:
- Agent 1: learns "near goal" predicate
- Agent 2: learns "obstacle detected" predicate
- Without sharing: each agent has partial information
- With sharing: agents benefit from each other's predicates

**Honest note**: We have not implemented symbolic knowledge transfer. The
DLR broadcasts require message-passing infrastructure.

### 4.3 Decentralized world model aggregation

Each agent `W_i` predicts next state. Aggregating predictions (e.g., via
median or learned attention) could provide better joint predictions.

This is similar to ensemble methods in supervised learning but applied
to dynamics prediction.

**Honest note**: This is a research direction, not a tested result.

---

## §5 Y2 experimental plan

### 5.1 Environments

We plan to test on 3 cooperative multi-agent benchmarks:

| Env | N agents | Joint reward | Notes |
|-----|----------|--------------|-------|
| PettingZoo Simple Spread | 3 | coverage-based | easiest |
| PettingZoo Simple Reference | 3 | coverage-based | medium |
| ParticleEnv cooperative navigation | 3 | distance-based | medium |

**Honest note**: These are 3 environments. We do not plan to test
StarCraft multi-agent or Hanabi (too complex for our compute budget).

### 5.2 Baselines

For each env, compare against:
- **Independent PPO**: each agent trains independently (no coordination)
- **Shared PPO**: parameter sharing across agents
- **QMIX**: standard cooperative MARL baseline

### 5.3 Hypothesis tests

**H2 (decentralized Monitors > joint Monitors)**: 5 seeds × 3 envs

If H2 holds in any env, we have a publishable result. If H2 fails
universally, we have a clear negative result.

### 5.4 What "successful" looks like

For H2:
- Y1 paper-like: 5 seeds, mean +X% over joint Monitors
- If X > 5%, publishable
- If X < 5% but positive trend, workshop paper
- If negative, negative-result paper

### 5.5 Compute budget

Multi-agent RL is more compute-intensive than single-agent:
- PettingZoo Simple Spread: ~30 min per seed (CPU)
- 3 envs × 5 seeds × 2 methods (decentralized vs joint) = 30 runs
- Total: ~15 hours wall time

This fits within Y2 budget if we use parallel seeds.

---

## §6 Limitations and honest unknowns

### 6.1 What we don't know

1. **Whether H2 even holds**: decoupling may not extend to multi-agent
   due to credit assignment differences.
2. **Whether DLR broadcasts help**: symbolic knowledge transfer is
   untested in multi-agent.
3. **Whether decentralized world models beat single-model**: aggregation
   effects are unclear.

### 6.2 What we haven't done

- Implemented any multi-agent code yet
- Run any multi-agent experiments
- Built the DLR broadcast protocol
- Aggregated world models

### 6.3 Risk factors

- **Compute**: multi-agent is more expensive. Without GPU, limited to
  small envs.
- **Environments**: PettingZoo benchmarks are easy; real-world multi-agent
  is much harder.
- **Credit assignment**: this is a fundamentally hard problem that
  decoupling may not solve.

---

## §7 Timeline (Y2 2027)

| Month | Task |
|-------|------|
| 2027-01 | Implement DMC architecture in code (no exp) |
| 2027-02 | PettingZoo Simple Spread baseline + DMC |
| 2027-03 | 3 envs × 5 seeds × 2 methods |
| 2027-04 | Cross-agent symbolic knowledge transfer |
| 2027-05 | Analysis + draft §1-3 |
| 2027-06 | Draft §4-6 + appendix |
| 2027-07 | Internal review + revisions |
| 2027-08 | Submit to AAMAS 2028 (deadline ~Sep 2027) |

---

## §8 Alternative outcomes

If Y2 results show:

1. **H2 holds strongly**: AAMAS submission + NeurIPS workshop
2. **H2 holds weakly**: Workshop paper (RLC, RLDM)
3. **H2 fails**: Negative-result paper, redirect to single-agent focus
4. **Compute unavailable**: Defer to Y3 or partner with GPU-equipped lab

---

## §9 Open questions

These are research questions we would explore in Phase 2:

1. **Decoupling vs joint in MARL**: does the H1 mechanism transfer?
2. **Cross-agent predicate transfer**: does sharing DLR predicates help?
3. **Decentralized world model aggregation**: does ensemble help?
4. **Credit assignment + decoupling**: how do they interact?
5. **Adversarial multi-agent**: does decoupling help in competitive settings?

---

*[End of Phase 2 outline. ~10 KB.]*
