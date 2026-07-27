# Discord / Reddit 长版公告 — DEC-0011 v0.4 HALT

> 2026-07-27
> Audience: AGI/RL researchers, my own future self, critique partners
> Format: long-form post for Discord, Reddit r/MachineLearning, or r/reinforcementlearning

---

## Headline

**Decoupled failure monitors work. Online gating does not. DEC-0011 v0.4 result.**

---

## The story

For the past month I've been running a 5-year AGI research program in public. The first major empirical claim was H1: "a decoupled Monitor (failure predictor) trained on a frozen policy gives better failure-detection than a Monitor trained jointly with the policy."

**Section 4.6-4.8 of my paper** tests H1: 5 seeds on LunarLander-v3, 100K PPO steps each. Frozen Monitor achieves AUROC mean = 0.796. Joint Monitor achieves AUROC mean = 0.072 (worse than random; learned the inverse signal). Delta = 0.724, 5/5 seeds support decoupling. **H1 strongly supported at the prediction level.**

The natural next question: if I have a good failure predictor, can I USE it to intervene? When the Monitor says "this trajectory is about to fail", switch the policy to a safer action.

**Section 4.10 (Phase 1.5)** is the 4-layer AGI integration demo. I tested the obvious intervention: Q-BoN (Best-of-N) where the Q-function is gated by the Monitor's failure probability. The Monitor says "high risk" → switch to argmax_Q over the action space, instead of PPO's action.

**It did not work.** Over 6 independent experiments across 2 environments, 3 action-selection strategies, and varying data sizes, I could not get a positive, statistically significant policy gain from the Monitor.

---

## The 6-way comparison (DEC-0011 v0.1 - v0.4C)

| # | Setup | n_train | n_eval | Delta | t-stat | Pos seeds |
|---|-------|---------|--------|-------|--------|-----------|
| v0.1 | Q-BoN, fixed threshold=0.5 (LunarLander) | 200 | 5 | +21.5 | 0.72 | 3/5 |
| v0.2 | Q-BoN, calibrated threshold (LunarLander) | 200 | 50 | -158.1 | -1.69 | 0/5 |
| v0.3 | safe action=2 (main engine), calibrated | 200 | 50 | -717.6 | **-3.71** | 0/5 |
| **v0.4A** | Q-BoN, calibrated, **1000 train episodes** | 1000 | 50 | **-1.8** | **-0.25** | 3/5 |
| v0.4B | Q-BoN, calibrated (CartPole-v1) | 200 | 50 | -270.4 | **-3.48** | 0/5 |
| v0.4C | Imitation learning (BC on top-25% PPO) | 200 | 50 | -33.7 | **-2.64** | 0/5 |

**0 out of 6 experiments show statistically significant HELP (positive delta with |t| > 2.78).** Five out of six are negative, four significantly so.

---

## What we learned

### 1. Data scale matters — a lot
v0.2 (200 train) failed because the val set (40 episodes, 4 failure cases) overfit to val_auroc=1.0, and the calibrated threshold collapsed to ~0 (Monitor always fires). v0.4A (1000 train, 200 val) had a sensible cal_threshold of 0.1-0.65 and gave a **neutral** delta. The 5x data increase turned a -158 into a -2.

But neutral is not positive. Even with 5x more data, the Monitor + Q-BoN does not beat PPO-only.

### 2. The Monitor''s signal is real but doesn''t prescribe action
The Monitor (SlotMonitor) achieves val_auroc 0.99 reliably across 6 experiments. It detects failure. But "failure is likely" is not the same as "this is the best recovery action." The action selection layer (Q-BoN, safe action, behavior cloning) is the bottleneck, not the Monitor.

### 3. The bottleneck is action-selection data
With 200 PPO rollouts (each ~500 steps = 100K transitions), neither Q-learning with CQL nor behavior cloning can learn a reliable action policy. The training data is just too small. Y1 work needs 10x+ more data (or fundamentally different action mechanisms).

### 4. The environment doesn''t change the conclusion
CartPole-v1 (a much simpler env where PPO scores 440-500 of 500 max) also fails with the same gating strategy (delta -270, t=-3.48). This is not a LunarLander-specific problem.

---

## What I''m NOT saying

I am **not** saying decoupled Monitors are useless. The Monitor itself (Sections 4.6-4.8) is a strong contribution: 0.99 AUROC, 5/5 seeds, p<0.01. The Monitor is also useful as:
- An offline diagnostic for understanding policy behavior
- A signal for human-in-the-loop interventions
- A training signal for imitation learning (the Y1 direction)

I am also not saying online gating is fundamentally broken. I tested 6 specific configurations. A more sophisticated action-selection mechanism (e.g., model-based planning, or using the Monitor only in a narrow "high confidence" regime) might work. But the easy wins are exhausted.

---

## DEC-0011 v0.4 final decision: HALT

I am stopping the online-gating sub-project. The decoupling contribution is **conceptual and prediction-level**, not policy-level. Further iterations would require:
- 10x+ more training data (1000+ PPO rollouts is not enough)
- Or a fundamentally different action mechanism (model-based, planning, or expert imitation at scale)
- Or a different environment where failure modes are more discrete

These are out of scope for Phase 1.5. They''re the Y1 problem.

---

## Y1 direction (next)

1. **Imitation learning at scale** (extension of v0.4C). Use a stronger expert (e.g., a hand-coded controller for LunarLander) and BC from that.
2. **Different env** (Acrobot-v1, MountainCar-v0). Smaller state space, more discrete failure modes.
3. **Monitor as training signal, not as gating signal**. Train PPO to avoid trajectories that the Monitor scores as "high failure" (regularization, not intervention).
4. **Model-based planning** (Project C). Use the world model to plan recovery actions, then gate only when planning confidence is high.

Code + paper: github.com/aidless/agi-research (MIT license, attribution 刘泽文)

Critique partners welcome: if you see a flaw in any of the 6 experiments, or have a suggestion for the Y1 direction, please reach out.

---

## Appendix: how to reproduce

Each experiment is 5 seeds, parallel background runs, ~10-30 min each on a single CPU.

```
python projects/project_a_self_improvement/code/full_integration_v2.py \
  --env LunarLander-v3 --n-ppo-steps 100000 --n-train-episodes 1000 \
  --n-eval-episodes 50 --target-fpr 0.10 --min-q-coverage 50 \
  --seed 0 --out-tag v4a --log-every 50
```

Per-seed JSON: `checkpoints/full_integration_v4a_LunarLander-v3_seed0/phase2_log.json`

Logs: `experiments_log/2026-07-27-phase15-v0p4-abc.md`

Paper: `projects/project_a_self_improvement/paper_v2_full.md` (Sections 4.6-4.8 for the supported H1, Sections 4.10-4.10.11 for the failed gating).
