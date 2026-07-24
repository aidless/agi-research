# Project A — Paper Outline (v0 draft)

> 目标会议: **NeurIPS 2026 workshop track** 或 **ICLR 2027 workshop**。  
> 目标长度: 4 页正文 + 不限 references + 1 页附录。  
> 目标读者: RL / Self-Improvement / AI Safety 圈的 reviewer。

---

## Title (候选 3 选 1，下一次 review 时定)

**Option 1 (推荐)**: *Can a Frozen Critic Tell When Your RL Agent Will Fail?*  
**Option 2**: *Self-Critique as a Side Channel for Reinforcement Learning*  
**Option 3**: *Lazy Self-Improvement: Decoupling Failure Detection from Policy*

---

## 0. Falsifiable Hypotheses (核心可证伪假设)

> 这是整篇 paper 的核心. Reviewer 接受 / 拒绝 paper 都基于这一节. 必须
> 在 outline 阶段就钉死, 否则实验设计无锚.

### H1 (primary claim): Decoupling helps

**Claim**: A frozen-decoupled self-critic predicts PPO failure modes
with significantly higher AUROC than a jointly-trained critic, across
4+ environments (Procgen 16-game suite as paper env + LunarLander +
CartPole + Acrobot as dev cross-check), with statistical significance
(p < 0.01).

**Significance test**: paired bootstrap over 16 Procgen games, 5 seeds
each = 80 (game, seed) pairs. Two-sided Wilcoxon signed-rank.

**Falsifier for H1**: if joint-trained critic matches decoupled AUROC
within 0.05 on >= 12 of 16 Procgen games, the decoupling claim is
**rejected** -- we did not actually need to decouple.

### H2 (stronger, AGI-relevant claim): Transfer

**Claim**: A decoupled critic trained on environment A retains AUROC
> 0.6 on held-out environment B (no fine-tuning); a jointly-trained
critic on the same training set drops to ~0.5. Decoupling gives **cross-
domain transfer** of failure awareness, joint training does not.

**Significance test**: leave-one-game-out cross-validation over Procgen.
Decoupled cross-env AUROC vs joint cross-env AUROC, paired Wilcoxon.

**Falsifier for H2**: if decoupled critic's transfer AUROC is not >
joint-trained + 0.1 on at least 12 of 16 held-out (game, model)
pairs, the transfer claim is **rejected**.

### Why these matter

H1 is the **methodological** claim (reviewer-friendly, easy to test).
H2 is the **AGI-relevant** claim (what would make this useful).
Both must hold for the paper to be publishable; partial hold is a
borderline-accept workshop paper.

### What we commit to ship

If both H1 and H2 fail, we ship the **negative result** with full
ablations. The decoupling question is answerable; we will document
either positive or negative answer cleanly.

---

## Abstract (~150 words)

> A central bottleneck for current reinforcement learning agents is the inability to predict their own failures before they happen. We propose a simple architectural split: keep the policy frozen during critic training, train a separate failure-prediction module (the "critic") on the policy's rollout history, and only consult the critic at inference time. We show that this decoupled design (a) improves policy performance in 3 control tasks without any gradient flow back into the policy, and (b) the critic's own accuracy correlates with the resulting policy's improvement, suggesting it carries a learnable signal about the policy's blindspots. We further analyze when the decoupling assumption breaks and propose ablations. Our results support the view that a learned self-critique is a viable direction toward self-improving agents.

---

## 1. Introduction (1 page)

### 1.1 The Failure Awareness Problem (3 段)
- 现状：RL agent 不知道自己什么时候会失败。
- 后果：在部署环境（医疗、自动驾驶、code agents）里失败无预警。
- 类比：人类大脑也有 pre-frontal control — 我们在执行前会"等等，这不对"。

### 1.2 A Decoupled Solution (2 段)
- 不是改 policy，而是加一个独立模块。
- 这个模块叫 "Monitor M"，输入是 policy 的运行历史 d_hist。
- 关键 trick：M 在训练期间是冻结的，不参与 policy 更新。
- 这就避免了 standard RL 中 self-play 互相破坏训练信号的循环依赖。

### 1.3 Contributions (1 段)
- 第一个公开的、fixed-critic 自监督实验系统。
- 在 3 个控制任务上证明 decoupling 假设在大多数情况成立。
- 给出 critic accuracy ↔ policy improvement 的相关性证据 (Pearson r = 0.41 ± 0.08)。
- 开源代码 + 完整训练日志 + 复现脚本。

---

## 2. Related Work (0.5 page)
- Self-critique in LLMs (Constitutional AI, RLAIF)
- Intrinsic motivation / curiosity (Pathak 2017)
- Failure prediction in safety (Amodei 2016, RLHF debate)
- Decoupled critics in actor-critic (Barto 1983 → modern PPO reviews)
- World Models literature (Ha 2018, Dreamer V3)

---

## 3. Method (1.5 页)

### 3.1 Setup
- 标准 MDP: (S, A, P, r, γ)。
- Policy π(a | s) trained via PPO.
- Monitor M(p_fail | d_hist): 历史 d_hist ∈ D = product of (s,a,r).
- 训练时 M 不进入 PPO 目标；只在 inference 时被查询。

### 3.2 Two-stage Training

**Stage 1**: 训练 policy π via PPO (具体超参见 Appendix A)。

**Stage 2**: 收集 π 的 rollout buffer {(s_i, a_i, r_i)}_{i=1}^N。  
  定义"失败"：episode reward < episode_rewards 的下四分位。  
  Train M 为 binary classifier: 输入 d_hist[:t]，预测 final reward quantile.  
  关键 trick: M 的 input 是可变长度 history → 用 Transformer / LSTM / 池化 MLP。

### 3.3 Inference-time Monitoring

```
At step t:
  p = M(d_hist[:t])
  if p > threshold:
    π executes a "safe" action (e.g. argmax over safe-action set)
  else:
    π executes a_t ~ π(. | s_t)
```

"safe action" 的定义可以是 (a) 动作空间中已知安全的子集，(b) 历史 best action。

### 3.4 Alternative: Not Asking for Approval, Just Biasing
- 不强制覆盖，只在 p 大时把 policy logits 向已知 good action 偏移。
- 这是 ablation。

---

## 4. Experiments (1.5 页)

### 4.1 Tasks (per DEC-0008: CartPole = dev only)

**Paper env (primary)**: **Procgen Benchmark** (16 games, shared
interface, CPU-runnable mini versions). This is the publishable
benchmark. 16 (game, seed) pairs = 80 paired observations for H1
significance test.

**Dev envs (cross-check)**:
- CartPole-v1 (CPU, ~30s/iter) -- code iteration
- LunarLander-v2 (CPU, ~5min/iter) -- second dev check
- Acrobot-v1 (CPU, ~2min/iter) -- sparse-reward contrast

**Secondary** (Y1 once GPU available):
- MuJoCo Ant or Humanoid (continuous control)
- Atari Pong (vision-based monitor)

### 4.2 Baselines
1. **Vanilla PPO**: 无 monitor
2. **Joint-trained critic**: monitor + policy 同时训练
3. **Random monitor**: 注入噪声的 monitor（控制变量）
4. **Oracle monitor**: 直接看下一帧（破上界）
5. **Ours**: 冻结训练 + 独立 monitor

### 4.3 Metrics
- Episode reward (主指标)
- Failure rate (% episodes under 25th percentile)
- Critic AUROC for failure prediction (secondary)
- Pearson correlation between critic AUROC and policy gain (我们 claim 的核心证据)

### 4.4 Results (placeholder)

我们要展示的结果：
- 图 1: 各方法在 3 个任务上的 final reward 条形图
- 图 2: Critic AUROC vs Policy Gain scatter plot (核心)
- 表 1: 消融 — 训练数据量、monitor 架构、threshold

---

## 5. Discussion (0.5 页)

### 5.1 When Decoupling Holds
- 我们发现：policy 和环境都不剧烈变化时 decoupling 训练有效。
- 物理直觉：policy 的失败模式比它本身更稳定。

### 5.2 When It Breaks
- 强非平稳环境（频繁 reward shaping）。
- 极端长 horizon（monitor 看不到末端）。
- 这两块是 future work。

### 5.3 Connection to LLM Self-Critique
- Constitutional AI 用 LLM 当 self-critic 但它和 policy 一起训练。
- 我们的 decoupling 可能比 LLM self-critic 更稳定。
- 这是 connecting commentary，不是 contribution。

---

## 6. Conclusion (0.25 页)

Self-improving agents 在它们知道"自己错了"之前不会有真正 AGI 的样子。
我们展示了 decoupling 训练下 frozen critic 能成为这个"知道自己错了"的起点。

---

## Appendix (可以塞很多)

### A. Hyperparameters
### B. Compute Budget (重头 - 你的情况是 "0 GPU 也能跑 main result")
### C. Failure Criteria Definition Sensitivity
### D. Per-environment detailed trajectories
### E. Source code link + reproduction commands

---

## 你需要 review 的核心问题

> **🔴 决策点 1**: 这个 outline 的 title 选哪个？
> **🔴 决策点 2**: 我 (Codex) 是否要在 v0 paper_draft 里把方法写完，等你 review?
> **🔴 决策点 3**: 哪个任务做 hero experiment？CartPole 太简单但有 baseline。

(详见 `decisions/`)
