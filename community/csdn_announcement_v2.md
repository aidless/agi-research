# CSDN 公告 — Archimedes AGI 5-year program + ENWI port 整合

> 2026-07-27
> 平台: CSDN
> 主题: 独立 5 年 AGI 研究计划：H1 ablation 通过 + 4-layer working code + ENWI 框架对接

## 标题
独立 5 年 AGI 研究计划：H1 ablation 通过 + 4-layer working code + ENWI 框架对接

## 标签
#AGI #强化学习 #自改进 #SlotAttention #ActiveInference #ENWI #开源 #PyTorch

## 正文

我从 2026-07-25 开始了一个独立的 5 年 AGI 研究计划（Archimedes
项目），今天已经 25+ commits，做一次中期报告。

**重大结果**：

### 1. H1 ablation 5/5 通过

```
5 seeds × 100K PPO each on LunarLander-v3
Frozen Monitor:  mean AUROC = 0.796
Joint Monitor:   mean AUROC = 0.072
Delta: 0.724 (5/5 seeds support decoupling)
```

解耦的 failure-prediction Monitor（基于冻结策略的 rollouts 训练）
远优于 joint 训练。**Decoupling 是 self-monitoring 的机制**。

### 2. Slot-Monitor 整合突破

```
Slot-attention as Monitor input:  AUROC 0.989
Raw-history as Monitor input:    AUROC 0.796
Delta: +0.193
```

Slot-attention 给 Monitor 提供了 structure-aware features（horizontal /
rotation / vertical / residual），比 raw flatten history 好 24%。

### 3. 4-layer AGI 整合（一个 run 里 all active）

```
[Sensors] -> [Slot World Model] -> [Monitor] -> [Q-BoN] -> [action -> env]
             + [Language Interface]
             + [LTL Verifier]
```

5 个 component 在 single run 里同时工作。代码在
`E:\agi-research\projects\project_a_self_improvement\code\full_integration.py`。

### 4. ENWI 框架对接

读了 F:\TMLR\Fusion\ENWI_PAPER.md（1482 行 AGI 主论文），
port 了 ENWI 的 4 个核心 component 到 E:\agi-research\：

- **4 Physics Modules**（Gravity/Collision/Friction/Inertia）→ Project C
- **Active Inference Engine**（free energy minimization）→ Project A
- **Differentiable Logic Reasoner**（fuzzy AND/OR/NOT/IMPLIES）→ Project E
- **Composable Physics** (Composer + gate net) → Project C

ENWI 的 5-layer 架构（Embodied → SSM → Encoders → Composable Physics →
DLR → AIE）比我们的 4-layer（Monitor + WM + Lang + Verifier）更完整。
我们的工作可以看作 ENWI 的子集实现。

### 5. Slot world model dynamics

```
next-step prediction error: 0.000007 (近 perfect)
```

slot-attention + per-slot MLP dynamics 学到了 LunarLander 的 1-step transition。

### 6. Honest negative results

- **TTC BoN+Monitor**: 7 attempts, all negative (gating doesn't reliably help PPO on LunarLander)
- **Composable vs Monolithic (smoke)**: composable 3.5x WORSE than monolithic
  (ENWI claims 94% improvement, not replicated at smoke level)
- **CartPole / MountainCar at 4K PPO**: all NaN (PPO too undertrained)

## 仓库

- **GitHub**: https://github.com/aidless/agi-research (public, 46+ commits)
- **Zhihu**: https://www.zhihu.com/pin/2064649194275714554 (previous announcement)

## 计划

- **Y0 Q4** (现在 → 2026-12): 跑 ENWI Prediction 2 with 2000 epochs
- **Y1 H1** (2027-01 → 2027-06): 跨环境 demo + main-conference 投稿
- **Y2+** (2027-2029): full ENWI 实现

## 引用

Liu Zewen (2026). Archimedes: A Self-Improving AGI Substrate.
Independent 5-year research program, AGI-2026-001.
github.com/aidless/agi-research