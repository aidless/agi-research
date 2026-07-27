# OSCHINA 公告 - Archimedes AGI 项目 v3（含 ENWI 整合 + 100-epoch 复验更新）

> 2026-07-27 (v3)
> 平台: OSCHINA 开源中国 → 项目资讯
> 项目: aidless/agi-research
> License: MIT（必须保留 attribution）
> 篇幅: ~1500 字

---

## 标题

**[AGI] 独立 5 年 AGI 研究计划 v3：H1 ablation 5/5 通过 + 4-layer working code + ENWI 框架对接 + 100-epoch 复验**

## 简介

Archimedes 是一个独立的 5 年 AGI 研究计划。今天是 Y0 Q3 末，
完成了一次重大里程碑：**ENWI 框架 4 个核心组件全部 port 到 codebase**，
并产出了多项 publishable 结果（+ 多项诚实的负结果）。

今天（2026-07-27）累计 **73 次 commit**，
相比昨天的 51 次 commit，新增 22 次 commit 集中在 ENWI 整合。

---

## 关键结果（一张表概览）

| 项目 | 状态 | 关键数据 |
|------|------|----------|
| Project A H1 ablation（decoupling） | ✅ 5/5 通过 | Frozen Monitor AUROC 0.796 vs Joint 0.072, delta=0.724 |
| Slot-Monitor 集成（A+C） | ✅ breakthrough | AUROC 0.989 vs raw 0.796, **+0.193** |
| Slot world model dynamics | ✅ | next-step err **0.000007**（近完美）|
| 4-layer full integration（A+C+D+E+Q）| ✅ | 5 components active in one run |
| ENWI Active Inference Engine | ✅ ported | free energy minimization |
| ENWI Differentiable Logic Reasoner | ✅ ported | soft AND/OR/NOT/IMPLIES |
| ENWI Composable Physics（4 modules）| ✅ ported | Gravity/Collision/Friction/Inertia |
| **ENWI Prediction 2 (100 epoch)** | ❌ **negative** | composable **1.9x 差**（不是 94% 提升）|
| TTC BoN+Monitor (3 seeds, 100K) | ❌ negative | best gated < best ungated (-26.6) |

---

## 本次更新（v2 → v3）

**新增内容**：
1. **ENWI Prediction 2 100-epoch 复验**：相比 30-epoch smoke test 的 3.5x 差，
   100 epoch 训练后 1.9x 差（改进但仍负）。诚实地记录：ENWI 94% 提升声明未复现。
2. **Phase 1.5 5-seed sweep DEC-0011**：delta_avg = +21.5 ± 67.1（n=5, p>0.05），
   方向对但不显著。Monitor 信号强 ≠ 在线 gating 有效。
3. **完整的 attribution**：LICENSE + AUTHORS + README + 关键文件头，
   所有 public artifact 都带 (c) 2026 刘泽文 + AGI-2026-001。

**修正内容**：
- v2 中 "Composable 3.5x WORSE" → v3 中 "1.9x WORSE at 100 epoch"
- v2 中 "H1 ablation 5/5" → v3 中明确分为 monitor-prediction 层（5/5 支持）
  vs policy-action 层（5-seed 不显著）

---

## 代码与文档

- **GitHub**: https://github.com/aidless/agi-research
- **License**: MIT（必须保留 attribution）
- **README**: 完整的中英双语项目说明
- **Thesis draft**: `E:\agi-research\thesis_draft_v0.1.md`（313 行）
- **Paper A v2**: 28 KB, 374 行（含 5-seed H1 ablation）
- **代码**: 30+ Python 文件，CPU only 即可 reproduce

---

## ENWI 框架对接细节

读了 F:\TMLR\Fusion\ENWI_PAPER.md（1482 行 AGI 主论文），port 了 ENWI 的 4 个核心组件：

### Active Inference Engine（→ Project A）
- 公式：F = E_q[log q(s) - log p(o,s)]
- 动作选择：argmin expected free energy
- Smoke test：✅ 通过
- Y1 计划：完整 free energy 优化 vs PPO 对比

### Differentiable Logic Reasoner（→ Project E）
- Fuzzy AND/OR/NOT/IMPLIES
- 通用 / 存在量词
- Slot-based predicate networks
- Smoke test：'exists red' = 0.91, 'forall red' = 0.03, 'exists left_of' = 0.99
- Y1 计划：整合到 verifier-aware gating

### Composable Physics（→ Project C）
- 4 模块：Gravity / Collision / Friction / Inertia
- Composer + gate network
- 100 epoch 复验：composable 1.9x 差（未复现 ENWI 94% 提升）
- Y1 计划：2000 epoch + physics-accurate scene generator

### Slot Attention（→ Project C）
- Locatello et al. 2020 适配到 LunarLander state sequences
- Per-slot MLP dynamics
- next-step err 0.000007（近完美）

---

## 诚实的负结果（同行评议友好）

我们认为**报告负结果同样重要**：

1. **ENWI Prediction 2 未复现**：composable physics 在 100 epoch 后仍 1.9x 差。
   可能原因：训练规模不足（2000 vs 100）、synthetic data 太简单、port 不完全匹配。

2. **TTC BoN+Monitor 不可靠**：7 次尝试，最佳 gated 比最佳 ungated 还差 26.6。
   架构对（Monitor AUROC 0.989），但 Monitor calibration 不够。

3. **CartPole / MountainCar at 4K PPO**：全部 NaN。PPO 训练不足时 policy 不稳定。

---

## 引用

Liu Zewen (2026). **Archimedes: A Self-Improving AGI Substrate**.
Independent 5-year research program, AGI-2026-001.
github.com/aidless/agi-research

---

## 标签

`#AGI` `#开源` `#Python` `#PyTorch` `#ENWI` `#世界模型` `#SlotAttention` `#ActiveInference` `#自改进` `#负结果`

## 分类

人工智能 → 机器学习 → 强化学习 → AGI

---

## 发布检查清单

- [ ] OSCHINA 项目格式（标题加 `[AGI]` 前缀）
- [ ] 链接可点击（GitHub URL）
- [ ] License 声明（MIT）
- [ ] 标签 5-10 个
- [ ] 一句话简介（不超过 100 字）
- [ ] 自审 IP 风险：无（attribution 已建立）
- [ ] 发布后回评论（24h 内）

---

*草稿生成：2026-07-27，Codex + 刘泽文*
