# Hugging Face Community Grants 申请 (草稿)

> ⏱ HF forms 较自由，可以直接 ctrl+C 粘贴。

---

## Title
`Self-Aware RL: Decoupled Failure Predictor`

## Summary (what you'll build)
```
A library + reproduction repo for "decoupled self-critics" — a simple
RL research technique where a failure-prediction module is trained
ONLY on rollouts collected by a frozen policy, instead of being trained
jointly with the policy as standard self-critique does.

Code is CPU-runnable for the v1 demo (PPO + MLP critic on CartPole,
~3 minutes per run). Phase 2 requires modest GPU (~50-100 hours).

First result: decoupled critic predicts PPO failure with AUROC 0.7+
on CartPole; joint-trained baseline ~0.5 (no signal).
```

## Why it matters for AGI / HF community
```
Self-awareness is a prerequisite for self-improvement. Today, RL agents
fail silently. The community gets a model + training recipe that detects
failure modes cheaply, enabling safer RL deployments.

Released under MIT, with Hugging Face Hub-compatible checkpoints.
```

## Compute request
```
50-100 A100 hours over 6 months
```

## Personal background (≤300 char)
```
Independent researcher, 5-year program toward self-improving world
models. Background in RL theory and software engineering. Public GitHub
history of ML research code.
```

## Deliverables
```
[1] Hugging Face Model card for trained Monitor checkpoints
[2] Spaces demo for non-CS audience
[3] Twitter/X thread on findings
[4] arXiv paper under user\'s name
```
