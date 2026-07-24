# Decision Record 0003 — Project A 主 claim

> Date: 2026-07-24  
> Status: DECIDED  
> Owner: you + Codex
> Decision in this session: Frozen-Policy Decoupling 是 paper v1 的主 claim

## 上下文
Paper v1 的核心创新点是什么。

## 决策
**Decoupled Critic (frozen policy)** 作为唯一主 claim.

## 验证要求
- 必须证明 AUROC > 0.6 on held-out episodes
- 必须证明 same architecture with joint-training gives AUROC ~ 0.5
- 必须证明 across 至少 3 seeds main claim is reproducible

## 跟进
- Code review 后再开 paper_draft_v1
- 加 ablation: 训练样本量, Monitor 架构, 历史长度

