# Project A: Self-Improvement Loop

> **Status**: PoC 阶段，最优先项目
> **核心 idea**: 给 RL agent 加一个"自我批判"模块，让它能在执行前预测自己会失败。
> **目标**: 2026 年 3 月前有一篇 4-page arXiv 投稿。

---

## 这个项目的研究问题（一句话）

> 一个神经网络 agent 能否学会"在执行一个 action 之前，判断这个 action 是否会让自己失败"？

## 它为什么重要（5 年纲领里）

| 缺口 | 谁解决它 |
|---|---|
| **当前 RL agent 不知道自己会失败** | 所有 SOTA 模型 |
| Frontier lab 没人做（因为不涨 accuracy）| — |
| 这是 Self-Improvement Loop 的最简形式 | **你** |

如果 paper 成功，这会是 5 年纲领第一块被立起来的"AGI 子模块"。

## 三个版本（逐步扩大）

### v1: CartPole 最小 demo（已经代码化）
- CartPole-v1 环境
- PPO baseline
- 一个轻量 monitor (MLP) 看 hidden state 预测失败概率
- 比对 baseline vs baseline+monitor 的 success rate
- 算力: 任何笔记本 CPU 一小时内
- 状态: ✅ 代码已写在 `code/`

### v2: Atari Pong + 视觉 monitor（Year 0 Q3 目标）
- 加入 CNN encoder
- monitor 看 CNN feature 预测下一帧失败概率
- 算力: 至少 100 GPU-hour
- 状态: ⏳ 待算力

### v3: MuJoCo + hierarchical monitor（Year 0 Q4 / Year 1 Q1 目标）
- 半精度状态输入
- monitor 分层（fast / slow）
- 算力: 至少 1k GPU-hour
- 状态: ⏳ 待算力或实验室

## 关键概念：什么是"自改进循环"

```
                ┌──────────────┐
                │   Policy π    │
                └───────┬───────┘
                        │ action a_t
                        ▼
                ┌──────────────┐
                │    Env        │
                └───────┬───────┘
       observation o_t  │
                       │
       ┌───────────────┘
       │
       ▼  d_hist (历史) 
┌─────────────────┐               ┌────────────────────┐
│   Monitor M     │               │   Rollout Buffer   │
│ 输入: d_hist     │─────────────▶│   收集 (o,a,r)    │
│ 输出: 失败概率 p │               │                    │
└─────────────────┘               └────────────────────┘
```

**关键点**: Monitor M 在训练期间不参与 policy 更新（M 是"冻结"的）；
只在 inference 阶段被查询。这允许 M 与 π 异步训练且不污染对方的目标。

## 当前文件清单

```
projects/project_a_self_improvement/
├── README.md (本文件)        ✅
├── paper_outline.md           ✅ 你 review 后开始写论文
├── paper_draft_v0.md          ⏳ 等 paper_outline review 完成
├── code/
│   ├── main.py                ✅ 主入口
│   ├── ppo.py                 ✅ PPO 实现
│   ├── monitor.py             ✅ Monitor 模块
│   ├── envs.py                ✅ 环境
│   ├── evaluate.py            ✅ 评估脚本
│   └── README.md              ✅ 怎么跑
├── experiments/                ⏳ 你跑过实验后存日志
├── figures/                    ⏳ 生成的图
├── notes/                      ⏳ 你的 review 笔记
└── references/                 ⏳ 引用追踪
```

## 你需要 review 的三个文件（按优先级）

| 优先级 | 文件 | 你 review 的问题 |
|---|---|---|
| 🔴 P1 | `paper_outline.md` | 这个 idea 站得住吗？ |
| 🔴 P1 | `code/main.py` | 这个逻辑跑得通吗？ |
| 🟡 P2 | `code/monitor.py` | monitor 的设计合理吗？ |

## 我（你）需要做的关键决策

详见 `decisions/0002-pa_first_environment.md` 和 `0003-pa_monitor_architecture.md`。

简要说：你要选 (a) CartPole / Pong / MuJoCo，哪个做第一实验。
