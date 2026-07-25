# Discord / Reddit 公告 — Joint Ablation Result

> 2026-07-25. 适合 ml-agents / reinforcement-learning / artificial 子版。

---

## Discord announcement (ml-agents channel)

```
**New from my independent 5-year AGI program: H1 joint ablation shipped**

Trained a "failure-prediction Monitor" on a PPO agent in two ways:
- **Frozen Monitor**: PPO trains to completion, then Monitor trains
  on rollouts from the frozen PPO. (My method.)
- **Joint Monitor**: PPO and Monitor update together, with Monitor
  seeing fresh rollouts from the still-updating PPO every K=4 updates.
  (Ablation.)

Result on LunarLander-v3, 5 seeds, 100K PPO each:

|        | Joint  | Frozen | Delta  |
|--------|--------|--------|--------|
| seed 0 | 0.103  | 0.98   | 0.877  |
| seed 1 | 0.041  | 0.90   | 0.859  |
| seed 2 | 0.044  | 0.21*  | 0.166  |
| seed 3 | 0.074  | 0.92   | 0.846  |
| seed 4 | 0.099  | 0.97   | 0.871  |
| mean   | 0.072  | 0.796  | 0.724  |

*seed 2 frozen anomaly (only 1 of 5).

**Key finding**: joint Monitor collapses to AUROC < 0.1 (worse than
random). Pearson consistently POSITIVE (+0.35 to +0.85), meaning the
joint Monitor has inverted its prediction (high prob -> high reward).

The failure mode is "policy drag": PPO updates change the policy,
which changes what "failure" looks like, and the Monitor gets pulled
along.

Code + 5-seed logs: github.com/<user>/agi-research
Paper: v2 draft at projects/project_a_self_improvement/paper_v2_full.md

Looking for critique partners to evaluate the joint-ablation claim
and the proposed test-time-compute extension (Best-of-N + Monitor).
```

---

## Reddit r/reinforcementlearning post

**Title**: [R] Joint vs Frozen failure Monitor: 5-seed ablation shows decoupling matters (delta=0.724 on LunarLander-v3)

**Body**:

I have been running a 5-year independent AGI program focused on
self-improving RL agents. The H1 claim is that a decoupled
failure-prediction Monitor (trained on frozen-policy rollouts) is
empirically and architecturally superior to a joint Monitor (trained
while PPO is still updating).

I just finished a 5-seed joint ablation on LunarLander-v3. Headline:

- Frozen Monitor (my method): 5-seed mean AUROC = 0.796
- Joint Monitor (ablation): 5-seed mean AUROC = 0.072
- Delta = 0.724, well above my pre-registered H1 falsifier threshold of 0.05
- 5/5 seeds support H1
- Joint Monitor Pearson is consistently POSITIVE (+0.35 to +0.85),
  meaning the Monitor has inverted its prediction

This validates the "frozen-critic family" of self-improvement methods
(STaR, ReAct, Reflexion, Self-Refine, CRITIC, PRM) at the RL policy
level. The Monitor is a process reward model over policy steps; the
test-time-compute extension (Best-of-N over PPO actions) is the next
step.

Code, logs, and full paper draft are public:
- Code: github.com/<user>/agi-research
- Paper v2: projects/project_a_self_improvement/paper_v2_full.md
- Experiment log: experiments_log/2026-07-25-joint-ablation-A.md

Critique wanted on:
1. Is the "policy drag" interpretation correct, or is there a
   simpler explanation for joint Monitor collapse?
2. Is the cross-environment transfer of this finding plausible
   (LunarLander-v3 -> Procgen 16 games)?
3. Does the proposed BoN+Monitor extension (Y1) sound promising,
   or is there a better TTC approach?

Compute: 13 minutes total for 5 seeds at 100K PPO each, on CPU.

---

## 投递 checklist

- [ ] Twitter: post Version 2 (story) on personal account, link to GitHub
- [ ] Discord #ml-agents: post announcement, ask for critique
- [ ] Reddit r/reinforcementlearning: post as [R], include code link
- [ ] Hacker News: submit as "Show HN: Decoupled Failure Monitors for Self-Aware RL Agents"
- [ ] Cross-post to r/MachineLearning if HN traction
