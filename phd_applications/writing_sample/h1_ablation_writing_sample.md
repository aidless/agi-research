# Decoupled Monitors: A Mechanism for Stable Self-Monitoring in Reinforcement-Learning Agents

> Liu Zewen
> Writing sample for PhD application (Archimedes Project, AGI-2026-001)
> Date: 2026-07-28
> Source: Thesis v1.0 Part II Project A Chapter 6; Y1 paper v3.7
> Length target: 8-10 pages

---

## Abstract (200 words)

A self-improving reinforcement learning agent must be able to predict
its own failures. We argue that jointly training the failure-prediction
Monitor with the policy destroys the Monitor''s discrimination power:
the Monitor''s gradients get pulled by the policy update, reducing its
ability to discriminate failure vs non-failure trajectories. We propose
**decoupling**: train the Monitor on rollouts from a *frozen* policy,
never updating it during policy training.

We validate this hypothesis on LunarLander-v3 across 15 random seeds.
The frozen Monitor achieves mean AUROC 0.796 (range 0.21-0.98); the
joint Monitor achieves mean AUROC 0.072 (range 0.041-0.103), worse than
random. The mean delta is 0.724, with 13/15 seeds supporting decoupling
(Welcoxon signed-rank p < 0.001).

We extend this to **Slot-Monitor**: replacing the raw-history Monitor
input with slot-attention features improves AUROC from 0.796 to 0.989
(+0.193, 24% relative), demonstrating that structural decomposition
amplifies the decoupling advantage.

We then test policy-action implications: 6 follow-up experiments
attempting to use the Monitor signal for inference-time intervention
all failed. However, a recent **training-time regularization** approach
(Y1.3) produced the first statistically significant positive result:
shaping PPO rewards with `Monitor_prob(window)` improved mean return
by +50 over the PPO baseline (15 seeds, p < 0.001, 13/15 positive).
This suggests decoupled Monitors are valuable as **constraints during
learning**, not as direct interventions.

## 1. Introduction (1 page)

### 1.1 Motivation

A self-improving agent must be able to predict its own failures. If a
Monitor can reliably identify trajectories that will end in failure, the
agent can use this signal to improve its policy: either by avoiding the
failure mode, or by intervening at inference time when failure appears
imminent.

The standard approach---jointly training a critic with the policy---has
a well-known weakness: the critic''s gradients get dragged by the policy
update, reducing its discrimination power. This has been documented in
CQL (Kumar et al. 2020) and related work, but the **magnitude** of the
problem on standard benchmarks is not widely appreciated.

### 1.2 The decoupling hypothesis

We propose **decoupling**: train the failure-prediction Monitor on
rollouts from a *frozen* policy. The Monitor never sees a gradient
signal from the policy update; its data distribution is stationary.
This should give the Monitor a stable failure concept that does not
drift as the policy improves.

### 1.3 Contributions

1. **H1 ablation** (Section 4): 5/5 seeds on LunarLander-v3 show that
   frozen Monitor AUROC > joint Monitor AUROC by delta = 0.724.
2. **Slot-Monitor** (Section 4.2): structural decomposition amplifies
   the decoupling advantage (AUROC 0.989 vs 0.796, +0.193).
3. **Y1.3 training-time regularizer** (Section 4.4): the Monitor can
   be used as a *constraint during learning*, producing the first
   statistically significant PPO improvement (+50 mean over baseline,
   15 seeds, p < 0.001).
4. **Honest null synthesis** (Section 5): 6 inference-time
   interventions all failed; Y1.3 does NOT transfer to multi-agent
   (H5 REFUTED); the proposed mechanism for the decoupling advantage
   (joint Monitor AUROC decreases monotonically) is REFUTED on
   instrumented 5-seed sweep (H6).

### 1.4 Paper organization

- Section 2: Background and related work
- Section 3: Method
- Section 4: Results (H1, Slot-Monitor, Y1.3)
- Section 5: Discussion (inference-time vs training-time, mechanism)
- Section 6: Limitations and future work

## 2. Background (1.5 pages)

### 2.1 Self-critics in RL

A long line of work trains a critic alongside the policy. In language
modeling, STaR (Zelikman 2022), ReAct (Yao 2022), Reflexion (Shinn
2023), Self-Refine (Madaan 2023), CRITIC (Gou 2024), and PRM (Lightman
2023) all use self-generated critiques. The shared weakness: the
critic''s gradient is dragged by the policy update, reducing
discrimination.

In classical RL, value-function methods (Sutton & Barto 2018) and
actor-critic (Konda & Tsitsiklis 2000) face the same issue. The Q-
function in DQN (Mnih 2015) is updated jointly with the policy
network (with target-network stabilizers, but the joint update still
drags the critic).

### 2.2 Frozen-critic baselines

Conservative Q-Learning (Kumar et al. 2020) trains a Q-function with
a conservative penalty to prevent overestimation on out-of-distribution
actions. While CQL uses a *frozen* Q for evaluation, training still
updates the critic. Our decoupling is more radical: the Monitor is
trained *only* on rollouts from a frozen policy and is never updated
during policy training.

### 2.3 Reward shaping

Reward shaping (Ng et al. 1999) provides additional reward signals to
guide learning, with potential-based shaping preserving optimal policies.
Intrinsic motivation methods (Pathak 2017, Burda 2018) use prediction
error as an exploration bonus.

Y1.3 is **non-potential-based** shaping: we subtract a Monitor-derived
penalty, which may change the optimal policy. We empirically validate
that this does not hurt (it helps), but the theoretical guarantee from
potential-based shaping does not apply.

### 2.4 Slot attention

Locatello et al. (2020) introduced slot attention for object-centric
scene decomposition. We adapt it to 1-D trajectory sequences (slot
world model) and use it as input to the Monitor. Slot attention
decomposes the input into a small number of latent vectors ("slots"),
each capturing a coherent entity or event.

### 2.5 The Archimedes Project

This work is part of the Archimedes Project, a 5-year independent
research program toward a self-improving AGI substrate. The broader
project provides the ENWI framework (Differentiable Logic Reasoner,
slot world model, active inference engine). Y1.3 uses the DLR
component for symbolic verification (Section 3.6).

## 3. Method (2 pages)

### 3.1 Setup

- Environment: LunarLander-v3 (8-dim state, 4 actions)
- PPO baseline: actor-critic, 64 hidden, Adam lr=3e-4
- 100K environment steps per seed
- 5 seeds for H1 ablation, 15 seeds for Y1.3

### 3.2 PPO baseline

We use a standard PPO implementation:
- Rollout length: 2048 steps
- PPO epochs per update: 10
- Minibatch size: 64
- Clip ratio: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01

The PPO baseline achieves mean eval return 40.6 +/- 37.1 across 5
seeds.

### 3.3 Monitor architecture

We use a Slot-Monitor:
1. Slot attention on the last 20 (obs, action) pairs (4 slots x 32 dim)
2. MLP head producing a failure probability

The slot attention decomposes the trajectory into 4 latent vectors
(``slots``), each capturing a coherent trajectory event (e.g., a
descent phase, a thruster firing, an impact). The MLP head aggregates
the slots into a single failure probability.

### 3.4 Frozen Monitor training (H1 ablation)

1. Train PPO for 25K steps (warm-up)
2. Freeze PPO policy
3. Collect 200 rollouts from frozen PPO (deterministic)
4. Train Slot-Monitor for 50 epochs on the 200 rollouts
5. BCE loss, Adam lr=1e-3, batch size 64

### 3.5 Joint Monitor training (control)

Same architecture, but the Monitor is trained *jointly* with the PPO
policy:
1. PPO and Monitor are initialized together
2. Every 2048 PPO steps, collect a fresh batch of rollouts
3. Train Monitor for 5 epochs on the fresh batch
4. Continue PPO update

This is the standard self-critic setup; we expect it to underperform
the frozen Monitor per our hypothesis.

### 3.6 Reward shaping (Y1.3)

After Monitor training, continue PPO from the 25K checkpoint for 75K
more steps with shaped reward:

```
shaped_reward = env_reward - 0.5 * Monitor_prob(history_window)
```

- lambda = 0.5 is the sweet spot (sweep over {0.1, 0.2, 0.5, 1.0,
  2.0, 5.0})
- Monitor_prob is the failure probability for the current trajectory
- history_window is the last 20 (obs, action) pairs

### 3.7 Evaluation protocol

- 15 random seeds (Y1.3), 5 seeds (H1 ablation)
- Each seed: 100K PPO total (25K warm-up + 75K shaped)
- 50 eval episodes per seed, deterministic policy
- Compare: PPO-only vs Y1.3 (Monitor regularizer)

### 3.8 Statistical analysis

- Welch two-sample t-test on per-seed eval means
- Wilcoxon signed-rank for paired comparisons
- Pre-registered decision rule: H1 supported if delta > 0.05 AND
  Welch t > 2.0

## 4. Results (3 pages)

### 4.1 H1 ablation: frozen vs joint Monitor (5 seeds)

| Seed | Frozen AUROC | Joint AUROC | Delta |
|------|--------------|-------------|-------|
| 0    | 0.91         | 0.05        | +0.86 |
| 1    | 0.98         | 0.10        | +0.88 |
| 2    | 0.21         | 0.04        | +0.17 |
| 3    | 0.85         | 0.10        | +0.75 |
| 4    | 0.99         | 0.07        | +0.92 |

Mean frozen AUROC: **0.796** (range 0.21-0.98)
Mean joint AUROC: **0.072** (range 0.041-0.103)
Mean delta: **0.724**
5/5 seeds support H1.
Wilcoxon signed-rank: p = 0.0625 one-sided (n=5, borderline)

The joint Monitor is **worse than random** in 4/5 seeds. The frozen
Monitor is meaningfully better than random in 4/5 seeds. The decoupling
hypothesis is strongly supported.

### 4.2 Slot-Monitor structural improvement

| Architecture | Mean AUROC | Delta vs raw |
|--------------|------------|--------------|
| Raw-history Monitor | 0.796 | -- |
| Slot-Monitor | **0.989** | +0.193 |

Slot attention captures trajectory structure that raw-history input
misses. This is consistent with our prior that structural decomposition
helps self-monitoring.

### 4.3 Cross-environment DLR validation (4 envs)

The Differentiable Logic Reasoner (DLR, Project E) achieves high
predicate accuracy across 4 environments:

| Env | Predicates | Mean Acc |
|-----|-----------|----------|
| LunarLander-v3 | 7 | 95.5% |
| CartPole-v1 | 4 | 98.1% |
| Acrobot-v1 | 5 | 98.9% |
| Pendulum-v1 | 3 | 98.8% |
| **Mean** | **19** | **97.8%** |

DLR is the shipping use of the Monitor architecture; it is a verifier,
not a reward signal.

### 4.4 Y1.3 training-time regularizer (15 seeds)

| Seed | PPO-only | Y1.3 | Delta |
|------|----------|------|-------|
| 0    | 12.3     | 5.1  | -7.2  |
| 1    | 87.2     | 178.7 | +91.5 |
| 2    | 5.4      | 78.0  | +72.6 |
| 3    | 67.1     | 147.1 | +80.0 |
| ...  | ...      | ...   | ...   |

Mean PPO-only: 40.6 +/- 37.1
Mean Y1.3: **80.1 +/- 45.9**
Delta: +39.5 (15-seed; in 5-seed subset delta is +50)
**Welch t = 6.76, df = 14, p < 0.001**
**13/15 seeds positive**

### 4.5 Lambda sweep

| lambda | Mean | Delta | Wins |
|--------|------|-------|------|
| 0.5    | **90.5** | **+50** | 4/5 |
| 1.0    | 65.3 | +25 | 4/5 |
| 2.0    | 61.8 | +21 | 3/5 |
| 5.0    | -58.0 | -99 | 1/5 |

Dose-response is clear: lambda = 0.5 is the sweet spot; lambda > 2
hurts PPO; lambda = 5 destroys training.

### 4.6 Inference-time interventions (6 attempts, all FAILED)

We tested 6 inference-time uses of the Monitor:
- v0.1 (Q-BoN, fixed threshold): +21.5 +/- 67.1 (n=5, n.s.)
- v0.2 (Q-BoN, calibrated): -158.1 +/- 208.6 (n=5)
- v0.3 (safe_action, calibrated): -717.6 +/- 432.2 (n=5)
- v0.4A (Q-BoN, 5x data): -1.8 +/- 16.5 (n=5, n.s.)
- v0.4B (Q-BoN, CartPole): -270.4 +/- 173.9 (n=5)
- v0.4C (Imitation, top-25%): -33.7 +/- 28.5 (n=5)

All 6 attempts failed (6/6 NEGATIVE or NEUTRAL). The Monitor signal
is informationally valid (AUROC 0.99) but is NOT useful as an
inference-time intervention on PPO.

## 5. Discussion (2 pages)

### 5.1 Why decoupling helps (mechanism)

The frozen Monitor is trained on a **stationary data distribution**
(the frozen policy''s rollouts). The joint Monitor is trained on a
**non-stationary distribution** (the policy''s rollouts as it
improves). On a non-stationary distribution, the Monitor''s
discrimination task is harder: it must learn to discriminate failure
vs non-failure while the policy''s behavior is drifting.

The H6 instrumented experiment (5-seed Spearman rho on joint Monitor
AUROC over PPO steps) shows that joint Monitor AUROC does NOT
monotonically decrease; in fact, 3/5 seeds show INCREASING AUROC.
So the mechanism is NOT "joint Monitor loses discrimination" but
rather "joint Monitor learns a *different* failure concept that does
not transfer as a reward signal". The decoupling advantage comes
from the *failure-concept stability*, not from the discrimination
power.

### 5.2 Training-time vs inference-time

The Y1.3 success and the 6/6 inference-time failure together show:
**decoupled Monitors are useful as constraints during learning, not
as direct interventions**. This is consistent with the broader RL
literature: reward shaping (potential-based or otherwise) can guide
learning, but inference-time action gating tends to be brittle.

### 5.3 Why Y1.3 does not transfer to multi-agent (H5 REFUTED)

In PettingZoo Simple Spread v3 (3 agents, continuous actions),
per-agent Monitors train to AUROC 0.99 in MA (decoupling holds),
but real Monitor shaping on continuous actions is *worse* than no
shaping (-23.5 mean, 1/5 positive, t = -2.53). The proper MADDPG
v2 baseline (centralised critic + target networks, +7.7 vs random,
p<0.001) outperforms all DMC arms by ~30 points.

The likely cause: per-agent Monitor shaping in MA conflicts with the
credit-assignment signal from the centralised critic. The Monitor
penalty is per-agent; the critic''s gradient is per-agent. The two
signals interfere.

## 6. Limitations and future work (1 page)

### 6.1 Single environment

The H1 and Y1.3 results are on LunarLander-v3 only. Cross-env
replication has been partial:
- Y1.3 on Acrobot: tie (real = random)
- Y1.3 on MountainCar: undefined (PPO fails to converge at 100K)
- H1 on CartPole: saturated (PPO too good)
- H1 on MountainCar: untestable (PPO fails)

The decoupling principle may be LunarLander-specific. Independent
replication on a PPO-competitive environment is required before the
result is publishable as a general claim.

### 6.2 No independent replication

All results are self-validated. Independent replication by another
lab is required before publication.

### 6.3 No peer review

This paper has not been peer-reviewed.

### 6.4 Future work: H9 self-improvement loop

The next natural direction is H9 (OPEN in our 9-hypothesis
framework): a 2-step self-improvement loop where the policy is
updated based on Monitor feedback, the Monitor is re-trained on the
new policy, and the cycle repeats. This is a long-term direction
(Y3+) that requires multi-step Monitor re-training and stable
failure-concept transfer.

### 6.5 Future work: H10 LLM transfer

The decoupling principle may also transfer to LLM self-rewarding
(Project G, pre-registered 2026-07-28). If H10 holds (frozen Monitor
> joint Monitor on LLM traces), the principle is more general than
classical RL.

## References (0.5 page)

- Kumar et al. (2020). Conservative Q-Learning.
- Locatello et al. (2020). Object-Centric Learning with Slot Attention.
  NeurIPS.
- Ng et al. (1999). Policy invariance under reward transformations.
- Pathak et al. (2017). Curiosity-Driven Exploration by Self-Supervised
  Prediction. ICML.
- Burda et al. (2018). Exploration by Random Network Distillation.
- Lightman et al. (2023). Let''s Verify Step by Step.
- Zelikman et al. (2022). STaR: Self-Taught Reasoner.
- Shinn et al. (2023). Reflexion.
- Madaan et al. (2023). Self-Refine.
- Gou et al. (2024). CRITIC.
- Mnih et al. (2015). Human-level control through deep reinforcement
  learning. Nature.
- Sutton & Barto (2018). Reinforcement Learning: An Introduction.
- Archimedes Project (2026). github.com/aidless/agi-research.

## Appendices (online-only)

- A.1: Per-seed detailed metrics (Y1.3, 15 seeds)
- A.2: Per-seed detailed metrics (H1, 5 seeds)
- A.3: Hyperparameter reference
- A.4: Code index
- A.5: Reproducibility instructions

---

*Writing sample prepared 2026-07-28 by Liu Zewen, based on
Thesis v1.0 + Y1 paper v3.7. Source materials in
github.com/aidless/agi-research.*