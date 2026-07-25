# Decision Record 0011 - Test-Time Compute (TTC) as Monitor Extension

> Date: 2026-07-25
> Status: **PROPOSED** (P3, deferred to Y1 Q2)
> Source: F:\TMLR corpus H/I synthesis, Section 4.5

## Proposal

After Paper A v1 ships (target ICLR Workshop on Self-Improving Systems
April 2026), evaluate whether Best-of-N policy sampling with the
decoupled Monitor as a per-sample scorer yields a meaningful TTC gain.

## Why this matters

The F:\TMLR corpus I01 (Test-Time Compute Scaling) establishes TTC as the
2025-2026 meta-trend. o1, DeepSeek-R1, and AlphaProof all build on
PRM-driven search. Our decoupled Monitor is a process reward model in
disguise (predicts per-step failure probability from hidden state).

Best-of-N + Monitor gating is the natural extension: at eval time,
sample N candidate actions from PPO, score each with Monitor, pick the
lowest-failure-probability one. This is a *free* TTC gain on top of the
frozen Monitor we already ship.

## Concrete scope (P3, not P1)

- Decision deadline: Y1 Q2 (after first paper accepted or rejected)
- Decision owner: user, with Codex analysis
- Decision metric: does BoN+Monitor improve mean return by >= 5% on
  at least 4 of 16 Procgen games, holding sample count constant?
- Falsifier: if no improvement, discard TTC extension; document
  negative result as Paper A v2 addendum

## What does NOT change

- Project A Monitor architecture remains frozen (no retraining)
- Phase 1/2 plans unchanged
- Paper A v1 stays the priority deliverable for ICLR submission

## Open questions for the user

- Is TTC extension a Y1 priority, or Y2+?
- Should we attempt TTC extension *before* Paper A submission (risky),
  or strictly after acceptance?
- Compute budget: BoN sampling at N=8 means 8x eval cost. OK?

## Why P3 not P1

TTC extension requires:
1. Paper A v1 accepted (or at minimum, drafted to submission quality)
2. Monitor validated on Procgen baseline (K2 KPI >= 0.7 AUROC)
3. Multi-seed Procgen results (currently 5 seeds LunarLander, 1 seed Procgen)

All three are Y1 milestones. P3 is appropriate; escalate if any arrive
sooner than expected.

## Linked references

- F:\TMLR\前沿研究_01_TestTimeCompute.md (this synthesis)
- F:\TMLR\LLM应用架构_03_Agent框架与自主决策.md (ReAct/Reflexion cousins)
- Lightman 2023 "Let'"'"'s Verify Step by Step" (PRM)
- Snell 2024 "Scaling LLM Test-Time Compute Optimally"
