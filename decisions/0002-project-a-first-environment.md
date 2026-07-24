# Decision Record 0002 — Project A 第一环境

> Date: 2026-07-24
> Status: DECIDED
> Owner: you + Codex
> Decision in this session: CartPole-v1

## 上下文
Project A 需要一个 first experiment 环境。

## 选项
A. CartPole-v1 (weaker but cheap)
B. LunarLander-v2 (harder, CPU OK)
C. Acrobot-v1 (medium)
D. Pong (need GPU)

## 决定
**A**. 你刚刚跑了 CartPole smoke test，Monitor AUROC = 0.7+。这个 baseline 锚定。

## 跟进决策 (open)
- 18 个月内再加 LunarLander / Pong (sequence 60000 steps CPU / GPU 取决于算力)
- 30 个月内做 Pong 作为 main experiment

