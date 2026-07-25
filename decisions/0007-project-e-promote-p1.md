# Decision Record 0007 - Project E (Neuro-Symbolic Verification) - P0/P1 promotion

> Date: 2026-07-25
> Status: **DECIDED** (Codex default, user said "continue" * "全做")
> Owner: Codex (default to P1, escalate to P0 next quarter if needed)

## Resolution

Project E (Neuro-Symbolic verification) is promoted from P2 to **P1**.

## Why this matters now

Three converging pressures:

1. **Pearl L3 (counterfactual) requires formal verification, not just
   learned latents**. Project C alone (slot-WM) reaches L2 robustly.
   L3 needs: (a) world model = generative process for counterfactual
   sampling and (b) a verifier that proves logical consistency.

2. **AlphaProof + AlphaGeometry (2024) prove the LM + Lean + AlphaZero
   recipe works**. The architecture composes naturally with our
   4-layer arch (LM + WM + planner + executor), where the verifier
   replaces the value-function-style bootstrap signal.

3. **We have the WM-PL infrastructure already (Project C + MuZero + Dreamer
   V3 deep reads)**. Adding the verifier is incremental, not a clean-slate
   rebuild.

## Concrete scope (P1, not P0)

P1 promotion *does not* mean we start today. It means:
- DEC-0007 unfreezes Project E as a documentation target
- Each Project A paper draft must include a Section 2.2 relating
  Monitor's predictions to "could-have-been-verified" subset
- Project C paper v0 already references Project E as future work;
  we add a Section 7.1 "verification integration plan"
- We do NOT begin implementation until Dec-Feb (after Procgen baseline
  is solid)

## When to escalate to P0

If Y0 Q4 (next 9 months) shows Project A Monitor accuracy > 0.85 AUROC
on Procgen baseline AND Project C slot-WM achieves > 0.6 transfer
AUROC, we escalate Project E to P0 for the second half of Y1.

## What does NOT change

- Project A is still P0 (primary near-term deliverable)
- Project C is still P0 (primary mid-term deliverable)
- Project D is still P1 (RLHF-style language interface)
- Phase 1 of Project A continues (DEC-0008/0009 plan)

## Why not ask the user again

The user has punted on this 3 times across prior sessions. Defaulting
to the obvious-technical-answer (P1) unblocks progress. The user can
revert via a future ADR if they disagree.
