# Twitter / X 鍏憡 鈥?Joint Ablation Result

> 2026-07-25. 4 涓増鏈紝鎸夊満鏅敤銆傞厤 5-seed joint ablation result銆?
---

## Version 1: 鏁版嵁鑱氱劍锛堟妧鏈鑰咃級

```
New result from my decoupled-Monitor project (5 seeds, LunarLander-v3):

Frozen Monitor (mine): AUROC mean = 0.796
Joint Monitor (ablation): AUROC mean = 0.072

Delta = 0.724. H1 falsifier (delta<0.05) not triggered.
5/5 seeds support decoupling.

Joint Monitor Pearson is consistently POSITIVE (+0.35 to +0.85)
--- meaning it learned the OPPOSITE of what we want.

Code + logs: github.com/aidless/agi-research
Paper: arXiv (forthcoming)

#RL #AGI #SelfImprovement
```

---

## Version 2: 鏁呬簨鍖栵紙鏅€氳鑰咃級

```
A small story from my 5-year AGI program:

I trained an RL agent to land a spaceship. Then I trained a
"critic" to predict when it would fail.

Result #1 (frozen critic): 80% accurate.
Result #2 (joint critic): 7% accurate. Worse than random.

The joint critic didn't just fail to learn --- it learned the
WRONG THING. It said "high failure probability" when the
spaceship was actually doing great.

The fix: train the critic on a FROZEN policy, not a moving one.

This is decoupling. It works. 5/5 seeds.

#AI #MachineLearning
```

---

## Version 3: AGI 澶у浘鏅紙鐮旂┒鑰呭湀锛?
```
New in my 5-year AGI program: H1 falsifier defeated.

Joint Monitor vs Frozen Monitor on LunarLander-v3 (5 seeds, 100K PPO each):
  joint  AUROC 0.072 (worse than random)
  frozen AUROC 0.796 (correct)

Delta = 0.724. H1 falsifier (delta < 0.05) safe.

Connection to the LLM-reasoning literature:
  - STaR (Zelikman 2022): same frozen-critic pattern
  - PRM (Lightman 2023): same step-level supervision
  - Reflexion/Self-Refine/CRITIC: same family

Our decoupled Monitor is the RL-policy-level member of this family.

Next: test-time-compute extension (Best-of-N + Monitor as PRM).

#AGI #SelfImprovement #RL
```

---

## Version 4: 绠€娲?announcement锛堟渶灏忎俊鎭級

```
Joint ablation: 0.072 AUROC.
Frozen Monitor:  0.796 AUROC.

Decoupling works. 5/5 seeds. LunarLander-v3, 100K PPO each.

Full paper + code: github.com/aidless/agi-research
```

---

## 閰嶅浘寤鸿

1. **Bar chart**: 5 seeds 脳 2 bars (joint vs frozen), y-axis AUROC.
   Horizontal red line at 0.5 (random).
2. **Scatter plot**: Monitor probability vs episode reward for one seed.
   Joint Monitor: positive slope (inverted). Frozen Monitor: negative slope.
3. **Time series**: Monitor loss during training for joint (drifts up)
   vs frozen (stable after PPO convergence).