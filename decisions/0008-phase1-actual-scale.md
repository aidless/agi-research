# Decision Record 0008 - Phase 1 actual scale

> Date: 2026-07-25
> Status: **DECIDED**
> Owner: Codex (user said "继续", interpreted as default to A)

## Decision

Phase 1 actual scale:
- 4 train games (smallest T1): **coinrun, bigfish, jumper, dodgeball**
- 1 seed per game (proof of concept; increase seeds after PoC)
- 50,000 env steps per run
- Total: 4 runs * ~110s each = ~7-8 min CPU

## Why this scale

Full Phase 1 design (25M * 5 seeds * 8 games = 1B steps) is **25 days CPU**.
That is non-starter for Y0 Q2. We pick the smallest scale that produces
actionable numbers: enough to compute per-game p30 threshold AND to ship
a real JSON result we can analyse.

Scale-up path:
  - Step 1 (this decision): 4 games * 1 seed * 50K steps
  - Step 2: same 4 games * 3 seeds * 200K steps (~1 hour)
  - Step 3 (final): all 8 train games * 3 seeds * 200K steps (~3 hours)

## When we re-decide

When Step 1 produces JSON files with non-trivial variance, we move
to Step 2 without re-decision.
