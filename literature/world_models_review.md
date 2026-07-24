# World Models (Ha & Schmidhuber 2018) — Codex 深度评述

> **Citation**: Ha, D., Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
> **Status**: 必读 #1。这是 NeurIPS 2018 Oral paper，开启了"world model"研究范式。
> **本文目的**: 把这篇 paper 从"读完了"提升到"能跟领域内同行讨论"。

---

## 1. 三句话版本（reviewers 用的电梯 pitch）

> **作者在一个神经网络上学会了环境的整个动力学，并把它当作一个**"环境代理"**在内部做规划，从而不用在真实环境做搜索就能跑通 CarRacing 和 Doom 等游戏。**

具体包含 3 个模块：(V) 视觉编码器把图像压成 latent；(M) 在 latent 上跑 RNN 记忆的预测器；(C) 一个简单的 policy 在 (V+M) 提供的"幻觉环境"里训练。M 和 V 不参与 policy 训练（"孤立"掉了），这是**关键 trick**——它让一个当时的科研人员不需要 GPU 集群也能跑实验。

## 2. 三大贡献

### Contribution #1: 三模块拆分 (V / M / C)
```
  Observation (帧) ─▶ V ─▶ z_t (latent)
                              │
                              ▼
                       h_t = RNN(h_{t-1}, z_{t-1}, a_{t-1})
                              │
                              ▼
                       ĥ_{t+1} ──▶ ẑ_{t+1} (下一帧 latent 预测)
                              │
                              ▼
       C policy 从 [z_t, h_t] 选 a_t
```

### Contribution #2: 在"幻觉"中训练 policy
不去真实环境采样，而是直接用 M 模拟整个 rollout，C 只在 M 生成的潜空间轨迹上更新。这是后续 Dreamer 系直接沿用的核心。

### Contribution #3: 真正跑通了 + 公开代码 + 实验新颖
作者给了**完整可复现的代码**（keras + tensorflow.js + 浏览器 demo）。这一点在深度强化学习圈是个例外。它的实验设置（CarRacing, Doom, VizDoom）今天看来简单，但当时跑通已经是大事。

## 3. 该 paper 主要贡献背后的"暗功夫"

读 paper 表面看不出来，但其实它做了几个很难的工程决策：
1. **VAE 训练是用和 RL 分离的流程**——V 在静态帧上先训练然后冻结，M 单独训练，C 单独训练。这避免了 RL 信号把 latent space 推爆。
2. **temperature parameter τ** 控制 VAE 隐空间的"假设误差"。τ 大 → 隐空间更"模糊"，RNN 更容易预测。τ 小 → 重建更锐利，但更难预测。这是**可调节的归纳偏置**。
3. **Reward prediction 不是直接从 latent 回归**——它是从 VAE 输出端做的。这让 model 学到"想要的是 pixel-level reward"，语义耦合比直接 regress 更好。

## 4. 这篇 paper 解决了什么 / 没解决什么

| 维度 | paper 解决了 | 没解决的 |
|---|---|---|
| 短期规划 | ✅ 完整解决（纯 latent rollout）| — |
| 长期规划 | 部分解决（+ RNN state）| 长 horizon 仍 drift（paper 自己也承认）|
| 多环境泛化 | ❌ | 每训练一个环境要重新训 V、M |
| 反事实推理 | ❌ | Pearl Level 3 完全没有 |
| 元认知 / Self-Improvement | ❌ | 没有"我发现错了"的机制 |
| 语言 grounding | ❌ | latent 完全不可语言查询 |
| 数据效率 | 较弱 | CarRacing 1000+ episode 才稳 |
| 多步 credit assignment | 部分（RNN hidden state 隐式）| 没显式做 delayed reward |

## 5. 与你的 AGI 主题的连接

把你之前提的"三层闭环"映射到 World Models：

```
World Models                        你的 AGI 架构
─────────────────────────           ──────────────────  
V (visual encoder)        ⊆         Sensor Tower
M (memory + predictor)    ⊆         世界模型（核心）
C (controller/policy)     ⊆         规划器 + 执行器 混合
(无)                      ⊆         Self-Improvement Loop (Project A)
(无)                      ⊆         语言接口 (Project D)
───────────────────────────────────────────────────────────
```

**World Models 占了中间两层，缺上面（Self-Improvement）和侧面（语言接口）**——这恰好是你 Year 1 Project A 和 Year 2-3 Project D 的空白。

## 6. 衍生 / 必读文献

读这篇 paper 之前你应该已经知道：
- Sutton & Barto 2nd Ed, 第 9, 16, 17 章（MDP, function approximation, policy gradient）

读这篇 paper 之后你应该立刻去读：
- **PlaNet (Hafner 2019)**: World Models 的 latent-only 版，完全不用真实 rollout
- **Dreamer V1**: PlaNet + 真实环境闭环 + actor-critic
- **Dreamer V2**: discrete latent（categorical），Atari 上首次超人类
- **Dreamer V3**: 把规模提到 Minecraft，是这篇 paper 的工程化巅峰

## 7. 还没回答的开放问题 — 这是你的研究方向

### Q1: τ 的选择是 hand-tune 还是 learned?
**研究价值**: 中。手调没问题，但 learnable τ 会自动适应任务复杂度。
**你的实验**: 用一个 PRMBO 或元学习 controller 动态调整 τ，看是否提升 sample efficiency。

### Q2: M 是 deterministic 还是 stochastic?
**研究价值**: 高。World Models 用 RNN 的 deterministic state；后续 Dreamer 加 VAE-style stochastic。这是 Stochastic RNN 演化的起点。
**你的实验**: 在 CausalWorld 数据集上比较 deterministic vs stochastic RNN 的反事实预测能力。这是 Project C 的起点。

### Q3: V/M/C 是否真的能完全分离训练？
**研究价值**: 高但难。paper 用了固定 V，但理论上端到端训练可能更优。
**你的实验**: 设计一个 ablation study，比较 joint-end-to-end vs three-stage 在 Dreamer V3 同样的 benchmark 上。

### Q4: 没有"策略崩溃检测"——这是 Project A 的原爆点
**研究价值**: 极高。paper 没说 policy 在什么情况下会失败，但你**读 paper 第 4 节就能感觉到**——C 在幻觉环境里训练会 exploit M 的弱点。
**你的实验**: 在 CarRacing 上同时跑 baseline + 加一个 external monitor，衡量 monitor 是否能预测失败。**这就是你的 Project A paper 的 idea。**

## 8. 复现它的最小代价

论文给的代码可跑（浏览器 demo），但你想做严肃的 research reproduction：

```bash
# 推荐路径:
# 1. 安装 pytorch + gymnasium
pip install torch torchvision gymnasium

# 2. 复现 V (VAE) — CPU 可跑
# 3. 复现 M (RNN) — CPU 可跑
# 4. 复现 C (linear policy) — CPU 可跑
# 总算力：CartPole ≤ 30 min CPU
```

**当前会话里**: `projects/project_a_self_improvement/code/` 我已经写了一个最小可跑 baseline（Project A 的 PoC）。

## 9. 这篇 paper 你需要亲自读一遍的 5 个理由

1. **作为基准线** — 后续几乎所有 world model paper 都跟它比较
2. **三模块拆分的接口仍然好用** — 7 年后没有任何 paper 取代它
3. **代码公开** — 你的复现成本为 0
4. **τ + RNN 状态的思想** — 后续 JEPA 还在调用
5. **它失败的方式** — 跟今天的 SOTA 模型失败方式惊人相似：**长 horizon drift + 反事实不会 + 不能跨环境**。这些就是你 5 年的研究问题。

---

## 我的总结（一句话）

> World Models 是 AGI 研究的"正确的初始近似"，但它的每个组件都有清晰的"已知缺口"。这些缺口分布与你 5 年研究纲领的四个项目**几乎完美映射**——这不是巧合，是因为它抓住了核心，但还没解决核心。

---

## 你的下一步

读完这篇 paper 后，请到 `00_daily/your_papers_you_read/2025-XX-XX-world-models-ha-schmidhuber.md` 写 3-5 句话。这件事不能外包。

然后开新对话跟我说："我读完了 World Models。开始 Project A 的代码。"
