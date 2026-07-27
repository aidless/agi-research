# H1 v2 MountainCar — confirms PPO doesn't converge at 100K

> Date: 2026-07-28
> Mode: Quick smoke test of H1 cross-env on MountainCar-v0
> Status: **Cannot test H1** — PPO doesn't converge at 100K steps
> Author: 刘泽文 (Liu Zewen) + Codex

## 1. What we tried

H1 cross-env MountainCar-v0 quick test (1 seed, 100K PPO, 200 episodes).

## 2. Result

```
Frozen Monitor val AUROC: NaN (all-positive dataset)
Joint Monitor val AUROC:  NaN (all-positive dataset)
```

**Reason**: MountainCar-v0 episodes all have reward < -150 (PPO never
reaches the goal in 500 steps), so `is_failure_episode` returns True
for ALL episodes. The dataset is 100% positive — Monitor predictions
are constant — AUROC is undefined.

## 3. Confirmation of PPO limitation

This confirms the same finding as the Y1.3 extend (Acrobot + MountainCar
results, commit 416d0d7): **MountainCar-v0 at 100K PPO does not converge**.

Y1.3 stuck at -200.0 ± 0.0 on MountainCar — same conclusion.

## 4. Conclusion

H1 cross-env MountainCar is **untestable** with default PPO. To properly
test H1 on MountainCar, we would need:
- Longer PPO training (1M steps)
- Reward shaping (e.g., dense reward for height)
- Or a different sparse-reward env (e.g., Pendulum, which has continuous
  reward from -16 to 0)

## 5. Honest negative result

This is a useful boundary finding: **MountainCar is not in the Y1 test set
because PPO at 100K doesn't converge**. Future work would need better
baselines.

## 6. Artifacts

- `experiments_log/_h1_v2_mountaincar_seed0_quick.txt` (partial output)
- PPO confirmed to not converge (cross-validates Y1.3 finding)
