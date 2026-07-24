# Google Cloud Research Credits 申请 (草稿)

> 这是给 google 的 narrative proposal。你 ctrl+C 粘贴到申请表单再补具体字段。
> ⏱ 申请时常 < 1 月一次，每次结果是 email。

---

## Proposal title (≤80 char)
`Independent Research Program: Self-Aware Reinforcement Learning Agents`

## Abstract (≤2000 char)
```
I am an independent researcher pursuing a 5-year program to make
reinforcement learning agents self-aware of their failure modes,
as a stepping stone toward self-improving general agents.

My current experiments show that a decoupled failure-prediction module
(a "critic" trained ONLY on a frozen policy's rollouts, not the policy
itself) predicts PPO failure with AUROC 0.7 on CartPole (CPU-only
preliminary results; full multi-task evaluation is the immediate next step).

The decoupling assumption is novel, testable, and small enough that
single-GPU compute is sufficient. Validating it on Atari and MuJoCo
benchmarks (Dreamer V3 / TD-MPC baselines) requires ~100 GPU-hours.

A $1000-2000 cloud credit grant covers this entire experiment budget.
All code and trained checkpoints will be released open-source with a
peer-reviewed paper.
```

## Research plan (≤3000 char)
```
Phase 1 (months 1-3, GPU: ~50h):
  - Train 6 PPO + 6 PPO+Monitor configurations on 3 control tasks
  - 3 random seeds per configuration
  - Output: arXiv preprint + GitHub release

Phase 2 (months 4-7, GPU: ~50h):
  - Atari Pong, Breakout, SeaQuest with vision encoder
  - Compare decoupled vs joint-trained critic
  - Output: ICLR/NeurIPS workshop submission

Phase 3 (months 8-12, GPU: ~100h):
  - MuJoCo Ant, Humanoid
  - Multi-task Monitor (one Monitor across all envs)
  - Output: NeurIPS main-conference submission

Compute requests are modest because:
  - The architecture is small (under 50M parameters)
  - CartPole-control tasks dominate Phase 1
  - Vision tasks use PreResNet-18 (Dreamer V3 reference)
```

## Compute budget (real)
```
Phase 1: 50 × A100-hour  ×  $2.80/h    = $140
Phase 2: 50 × A100-hour  ×  $2.80/h    = $140
Phase 3: 100 × A100-hour ×  $2.80/h    = $280
Total request: $500-1000 in cloud credit
```

## Why independent? (honest, ≤500 char)
```
No university affiliation. All funding I have is from personal savings
and freelance ML engineering work. The goal of this program is not a
degree or commercial product; it is the satisfaction of working on
a question I believe is fundamental to AGI: can a system notice when
its own prediction will fail?
```

## Deliverables you commit to
```
[1] Public GitHub repo with all code, checkpoints, training logs
[2] arXiv submission within 6 months
[3] Conference (NeurIPS/ICML/ICLR) submission within 12 months
[4] Public blog post explaining the work in plain language
[5] Acknowledgement of GCP support in any publication
```
