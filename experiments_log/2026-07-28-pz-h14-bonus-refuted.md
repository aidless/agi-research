# H1.4 - Monitor as Exploration Bonus (Y1.3 vs Y1.4)

> Date: 2026-07-28
> Mode: PPO + Monitor as exploration bonus (different from Y1.3 reward shaping)
> Status: REFUTED - random bonus beats real Monitor bonus (-25.6)
> Author: Liu Zewen + Codex
> Pre-reg: 2026-07-28-PRE-REGISTERED-H1.4-v1.md

## 1. Pre-registered claim

H1.4 (per 2026-07-28-PRE-REGISTERED-H1.4-v1.md) tested a DIFFERENT
use of the Monitor than Y1.3:

- Y1.3 = Monitor as a training-time reward penalty
  (r_total = r_env - lambda * monitor_prob)
- Y1.4 = Monitor as an exploration bonus on the policy entropy
  (policy_bonus = monitor_prob)

Rationale: maybe shaping distorts the policy but a soft
exploration signal would not.

## 2. Setup

- Env: LunarLander-v3, 100K PPO steps, 25K warm-up, 200 Monitor
  training rollouts, 50 eval episodes.
- 5 random seeds for each of two arms:
  - REAL: real trained Slot-Monitor (slot-attention + BCE head,
    trained on rollouts from a frozen PPO policy).
  - RANDOM: monitor_prob ~ U[0,1] (control).
- Both arms add policy_bonus to PPO; otherwise identical.

## 3. Results (5 seeds per arm)

| arm | n | mean-of-means | sd-of-means | mean eval-std |
|---|---|---|---|---|
| PPO baseline (no Monitor) | 5 | 40.6 | 37.1 | 58.0 |
| H1.4 REAL | 5 | 52.7 | 24.0 | 53.2 |
| H1.4 RANDOM | 5 | 78.3 | 45.4 | 67.8 |
| Y1.3 (training-time penalty) | 15 | 80.1 | 45.9 | - |

Per-seed REAL - RANDOM deltas:
+41.87, -36.25, -0.84, -94.89, -37.93

Positive seeds (REAL > RANDOM): 1/5

Welch two-sample t (REAL vs RANDOM, equal-variance not assumed):
- t = -1.115, df ~ 6.1, SE = 22.96
- |t| < 2.776 -> not significant at alpha=0.05 (df=8 reference)

## 4. Honest interpretation

- REAL bonus is NOT better than RANDOM bonus; if anything the
  trend runs the wrong way (REAL is 25.6 lower on average).
- This contradicts the pre-registered expectation that a trained
  failure predictor would guide exploration better than uniform noise.
- Y1.3 (training-time penalty) still has the highest mean (80.1)
  and is the only Monitor use that beats PPO baseline robustly
  (15 seeds, t=6.76, p<0.001).
- H1.4 joins the REFUTED list alongside:
  - H2 cross-env: Acrobot tie, MountainCar undefined
  - H3 500K: Monitor signal HURTS (delta=-53.1)
  - All 6 inference-time interventions (DEC-0011)

## 5. Why might H1.4 fail?

- Bonus is too weak: a probability added to entropy has limited
  effect on the policy gradient compared to a reward penalty.
- Monitor is overconfident on PPO states: with 200 training
  rollouts the BCE-trained Monitor is biased toward PPO's
  failure modes, so it does not add new information.
- Random noise has higher variance (78.3 mean, 45.4 sd) and
  some seeds (e.g. seed 4) benefit from extra stochasticity.
- No early stopping / lambda tuning for the bonus.

These are POST-HOC explanations; they were NOT pre-registered.

## 6. What this means for the Y1 paper

- The only Monitor-on-RL intervention that consistently helps is
  Y1.3 (training-time penalty on the env reward).
- Y1.3 is NOT a generic Monitors help claim - it is specifically
  a reward penalty. Exploration bonus, joint training,
  inference-time intervention, and longer training (H3) all fail.
- This sharpens the Y1 paper claim: it is not Monitors help but
  decoupled Monitor as a reward penalty helps; nothing else works.
  Negative framing is a stronger contribution than positive-only
  reporting.

## 7. Action items

- [x] Update papers/y1_9hypothesis_framework.md H1.4 -> REFUTED.
- [x] Add H1.4 (bonus) REFUTED cross-reference in the Y1 paper draft.
- [x] Note in PROGRESS.md / H_ROADMAP that Monitor use is restricted
      to Y1.3 reward shaping.
- [ ] Optional follow-up: pre-register H1.4b with lambda=0.05
      (much smaller bonus) to rule out the too strong explanation.

## 8. Artifacts

- 10 phase2_log.json in
  projects/project_a_self_improvement/code/checkpoints/full_integration_y14*_LunarLander-v3_seed*
- 5 PPO baseline phase2_log.json in full_integration_ppobase_*
- Y1.3 reference: 15 phase2_log.json in full_integration_y13_*
- Launcher: experiments_log/_run_h14_5seed.ps1
- Done marker: experiments_log/_h14_5seed_20260728-125010.done

## 9. Lessons

- A pre-registered H1.4 lets us publish a negative result with
  full credibility (the protocol was fixed before seeing data).
- Negative results are first-class Y1 paper material; do not hide.
- The Monitor architecture is useful for verification (H3, DLR),
  not as a generic RL component. Restate the claim accordingly.
