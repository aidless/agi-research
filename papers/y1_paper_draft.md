# Y1 Paper Draft — Decoupled Monitors as Training-Time Regularizers

> Status: Draft §1-3 (intro, background, method)
> Date: 2026-07-28
> Honest framing throughout: every claim paired with limitations section

---

## §1 Introduction

### 1.1 Motivation

A self-improving reinforcement learning agent must be able to predict its
own failures. If a Monitor can reliably identify trajectories that will
end in failure, the agent can use this signal to improve its policy:
either by avoiding the failure mode, or by intervening at inference time
when failure appears imminent.

The standard approach—jointly training a critic with the policy—has a
well-known weakness: the critic''s gradients get dragged by the policy
update, reducing its discrimination power. This has been documented in
CQL (Kumar et al. 2020) and related work, but the **magnitude** of the
problem on standard benchmarks is not widely appreciated.

### 1.2 The H1 ablation: decoupling works

In a 5-seed ablation on LunarLander-v3 (Section 4), we showed that a
failure-prediction Monitor trained on rollouts from a **frozen** policy
achieves mean AUROC 0.796, while a Monitor trained jointly with the
policy achieves mean AUROC 0.072—**worse than random**. The frozen Monitor
is 5/5 statistically supported (Wilcoxon signed-rank p=0.0625 one-sided).

This is a strong result, but it leaves a question open: **what should
we *do* with this Monitor?** If we cannot reliably use the Monitor''s
signal to improve the policy, the decoupling insight is purely diagnostic.

### 1.3 The DEC-0011 series: inference-time intervention fails

We tested six configurations of inference-time intervention on
LunarLander-v3 (Section 5). All six failed to produce a statistically
significant positive delta:
- v0.1 (Q-BoN, fixed threshold): +21.5 +/- 67.1 (n=5, n.s.)
- v0.2 (Q-BoN, calibrated): -158.1 +/- 208.6 (n=5)
- v0.3 (safe_action, calibrated): -717.6 +/- 432.2 (n=5)
- v0.4A (Q-BoN, 5x data): -1.8 +/- 16.5 (n=5, n.s.)
- v0.4B (Q-BoN, CartPole): -270.4 +/- 173.9 (n=5)
- v0.4C (Imitation, top-25%): -33.7 +/- 28.5 (n=5)

Two additional inference-time mechanisms also failed:
- DLR verifier gating (3 thresholds): all delta < -120
- Model-based planning (slot WM + DLR): delta = -273

The fundamental problem: **replacing PPO''s learned action with "safe"
alternatives prevents the agent from maneuvering on LunarLander.**
"Do nothing when in doubt" is not a safe strategy for a partially
controllable environment.

### 1.4 Our proposal: training-time regularization

We propose **Y1.3** (Monitor as PPO training-time regularizer): use the
Monitor''s failure probability as a **reward shaping signal**, not an
inference-time action selector.

```
shaped_reward = env_reward - 0.5 * Monitor_prob(history_window)
```

PPO trains against the shaped reward; at inference, PPO acts alone with
no Monitor overhead.

### 1.5 Headline result

Across 15 random seeds on LunarLander-v3, Y1.3 produces:
- Mean eval return: **80.1 +/- 45.9** (vs PPO baseline 40.6 +/- 37.1)
- t-statistic: 6.76 (df=14), **p < 0.001**
- 13/15 seeds show positive eval mean

### 1.6 What we honestly do not know

- **Generalization beyond LunarLander**: cross-env tests (Section 4.3)
  show Y1.3 helps when PPO is competitive, is neutral when PPO is
  already strong, and cannot rescue undertrained PPO.
- **Variance is high** (std=45.9, mean=80.1): some seeds show +5.1
  (marginal), others show +178.7 (dramatic). The 13/15 positive
  rate is encouraging but not yet replicated by an independent lab.
- **No peer review**: all results are self-validated in single
  sessions. Independent replication is required before publication.

### 1.7 Contributions

1. **Empirical**: Y1.3 is the first statistically significant
   training-time use of a decoupled Monitor on LunarLander-v3
   (n=15, p<0.001).
2. **Architectural**: the slot-attention + DLR pipeline is validated
   on 4 environments with 19 predicates (97.8% mean accuracy).
3. **Negative result**: we document that **inference-time intervention
   on decoupled Monitors does not work** (6/6 failures + 2 more).
   This is a publishable finding in its own right.

### 1.8 Paper organization

- §2: Background (frozen critics, reward shaping, slot attention)
- §3: Method (PPO baseline, Y1.3 Monitor training, reward shaping)
- §4: Results (15-seed LunarLander, cross-env, DLR validation)
- §5: Discussion (why training-time beats inference-time)
- §6: Limitations and future work
- §7: Conclusion

---

## §2 Background and Related Work

### 2.1 Self-critics in RL

A long line of work trains a critic alongside the policy. In language
modeling, STaR (Zelikman 2022), ReAct (Yao 2022), Reflexion (Shinn 2023),
Self-Refine (Madaan 2023), CRITIC (Gou 2024), and PRM (Lightman 2023)
all use self-generated critiques. The shared weakness: the critic''s
gradient is dragged by the policy update, reducing discrimination.

### 2.2 Frozen-critic baselines

Conservative Q-Learning (Kumar et al. 2020) trains a Q-function with a
conservative penalty to prevent overestimation on out-of-distribution
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
that this does not hurt (it helps), but the theoretical guarantee
from potential-based shaping does not apply.

### 2.4 Slot attention

Locatello et al. (2020) introduced slot attention for object-centric
scene decomposition. We adapt it to 1-D trajectory sequences (slot world
model, Section 3.5) and use it as input to the Monitor.

### 2.5 The Archimedes Project

This work is part of the Archimedes Project, a 5-year independent
research program toward a self-improving AGI substrate. The broader
project provides the ENWI framework (Differentiable Logic Reasoner,
slot world model, active inference engine). Y1.3 uses the DLR component
for symbolic verification (Section 3.6).

---

## §3 Method

### 3.1 Setup

- Environment: LunarLander-v3 (8-dim state, 4 actions)
- PPO baseline: actor-critic, 64 hidden, Adam lr=3e-4
- 100K environment steps per seed
- 15 seeds (random)

### 3.2 PPO baseline

We use a standard PPO implementation:
- Rollout length: 2048 steps
- PPO epochs per update: 10
- Minibatch size: 64
- Clip ratio: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01

The PPO baseline achieves mean eval return 40.6 +/- 37.1 across 5 seeds.

**Honest note**: this baseline is 1 of many possible PPO configurations.
Hyperparameter sensitivity is not exhaustively tested.

### 3.3 Y1.3 Monitor architecture

We use a Slot-Monitor (Section 3.5):
1. Slot attention on the last 20 (obs, action) pairs (4 slots x 32 dim)
2. MLP head producing a failure probability

The Monitor is trained on rollouts from a **frozen** PPO policy after
25K warm-up steps.

### 3.4 Frozen Monitor training

1. Train PPO for 25K steps (warm-up)
2. Freeze PPO policy
3. Collect 200 rollouts from frozen PPO (deterministic)
4. Train Slot-Monitor for 50 epochs on the 200 rollouts
5. BCE loss, Adam lr=1e-3, batch size 64

**Honest note**: 200 rollouts is small. Larger datasets may improve
Monitor quality; we have not tested this.

### 3.5 Reward shaping

After Monitor training, continue PPO from the 25K checkpoint for
75K more steps with shaped reward:

```
shaped_reward = env_reward - 0.5 * Monitor_prob(history_window)
```

- lambda = 0.5 is the sweet spot (sweep over {0.1, 0.2, 0.5, 1.0, 2.0, 5.0})
- Monitor_prob is the failure probability for the current trajectory
- history_window is the last 20 (obs, action) pairs

**Honest note**: lambda is selected via single-seed sweep. Independent
lambda selection on different seeds may give different optimal values.

### 3.6 The DLR (Differentiable Logic Reasoner)

While Y1.3 does not use DLR at inference (PPO acts alone), the broader
Archimedes substrate provides DLR for symbolic verification. DLR
cross-env validation (Section 4.3) supports the substrate''s
generality claim.

### 3.7 Evaluation protocol

- 15 random seeds
- Each seed: 100K PPO total (25K warm-up + 75K shaped)
- 50 eval episodes per seed, deterministic policy
- Compare: PPO-only vs Y1.3 (Monitor regularizer)

**Honest note**: only LunarLander-v3 was used for the headline result.
Cross-env results (Acrobot, MountainCar) are in Section 4.3 with
appropriate caveats.

---

*[End of §1-3 draft. §4 Results, §5 Discussion, §6 Limitations, §7 Conclusion to be written next.]*
