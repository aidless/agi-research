# Decision Record 0009 - Phase 1 vs Monitor phase ordering

> Date: 2026-07-25
> Status: **DECIDED**
> Owner: Codex (user "继续" -> default A)

## Decision

**Phase 1 first** (policy-only baseline), then **Phase 2** (Monitor added).

## Why

Two-step shipping reduces risk:
- Step 1 yields per-game thresholds that calibrate Monitor failures
- Step 2 uses thresholds from Step 1 as input to Monitor training
- Together they cover the full Project A H1/H2 hypothesis pipeline

If we ship Phase 1+2 together, we cannot tell which component failed
when something goes wrong. The risk of wasted compute is too high.

## Output of Phase 1

Per (game, seed):
- episode_returns distribution
- p30 threshold (failure level)
- mean / median / std
- elapsed seconds

After Step 1 we know whether Procgen at this scale is even tractable.

## Output of Phase 2 (next session)

Same per-game runs but with Monitor training + evaluation. The Monitor
becomes the test of H1.

## Next action this session

Run the DEC-0008 Step 1 smoke right now. Commit JSON results.
