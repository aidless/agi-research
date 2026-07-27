# OSCHINA 公告 — Archimedes AGI 项目 v2（含 ENWI 集成）

> 2026-07-27
> 平台: OSCHINA 开源中国
> 项目: aidless/agi-research
> License: MIT

## 标题
[AGI] 独立 5 年 AGI 研究计划 v2：H1 ablation 5/5 通过 + 4-layer working code + ENWI 框架对接

## 简介

Archimedes 是一个独立的 5 年 AGI 研究计划。今天是 Y0 Q3 末期，
完成了一次重大里程碑：4 个 ENWI 核心 component 全部 port 到我们的
codebase，并产出了多项 publishable 结果。

## 关键结果

| 项目 | 状态 | 关键数据 |
|---|---|---|
| Project A H1 ablation | ✅ 5/5 通过 | Frozen Monitor 0.796 vs Joint 0.072, delta=0.724 |
| Slot-Monitor 整合 | ✅ | AUROC 0.989 vs raw 0.796, +0.193 |
| Slot world model | ✅ | next-step err 0.000007 |
| 4-layer full integration | ✅ | 5 components active in one run |
| ENWI 4 physics modules | ✅ ported | Gravity/Collision/Friction/Inertia |
| ENWI Active Inference Engine | ✅ ported | Free energy minimization |
| ENWI Differentiable Logic | ✅ ported | Soft AND/OR/NOT/IMPLIES |
| TTC BoN+Monitor (3 seeds) | ❌ negative | Best: -26.6 in favor of ungated |
| ENWI Prediction 2 (smoke) | ⚠️ | Composable 3.5x worse than monolithic |

## 仓库
- **GitHub**: https://github.com/aidless/agi-research
- **License**: MIT（必须署名）

## 引用
Liu Zewen (2026). Archimedes: A Self-Improving AGI Substrate.
Independent 5-year research program, AGI-2026-001.

## 标签
#AGI #开源 #Python #PyTorch #ENWI #世界模型 #SlotAttention