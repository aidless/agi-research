# Decision Record 0008 - CartPole as dev-only, Procgen as paper env

> Date: 2026-07-25
> Status: **DECIDED** (user confirmed 2026-07-25)
> Re-opens: DEC-002
> Owner: user + Codex

## Resolution

CartPole-v1 stays as the **development environment** for code iteration
(<30s per training run, CPU-only). The **paper-grade evaluation**
environment is **Procgen Benchmark** (16 games, shared interface,
CPU-runnable mini versions). LunarLander-v2 and Acrobot-v1 are kept
as secondary dev cross-checks.

## Why this is the right split

CartPole is solved by DQN in ~50 episodes. Any claim about general
intelligence or self-monitoring derived only from CartPole cannot be
argued at a publishable venue. But CartPole is still the fastest way
to iterate on Monitor architecture and hyperparameters. So we use it
**for development**, not for paper claims.

Procgen was designed exactly for cross-game generalization with shared
interface; 16 games x 5 seeds = 80 (game, seed) pairs is enough for a
Wilcoxon signed-rank test on AUROC.

## What changes downstream

- Project A paper outline Section 4.1 Tasks now lists Procgen as
  paper env (see `paper_outline.md` Section 4.1)
- Project A paper outline Section 0 (H1/H2 hypotheses) is committed
- DEC-002 resolution replaced by this record
