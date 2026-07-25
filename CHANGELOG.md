# CHANGELOG.md

> All material changes to TASKBOOK_v1.md are recorded here. Taskbook
> edits and ADRs stay in this file so v1.0 remains a stable baseline.

---

## v1.0 -> v1.1 (2026-07-25)

User feedback (5 critiques) drove 4 amendments. Each is recorded as
an ADR. TASKBOOK_v1 retains v1.0 text; this CHANGELOG is the
authoritative record of v1.1 modifications.

### Amendment 1: CartPole is dev-env only
- **Old (v1.0)**: CartPole-v1 = Project A first env (DEC-002 resolved)
- **New (v1.1)**: CartPole-v1 = DEV environment only. Paper-grade
  environment = **Procgen** (16-game benchmark). LunarLander +
  Acrobot as secondary dev cross-checks.
- **Why**: CartPole is solved by DQN in ~50 episodes; cannot support
  publishable claims about general intelligence or self-monitoring.
- **ADR**: decisions/0008-cartpole-as-dev-only.md
- **Status**: **DECIDED 2026-07-25** (user confirmed)

### Amendment 2: Y0 Q2 deliverable redefined
- **Old (v1.0)**: "1 arXiv paper out" by Q2 end
- **New (v1.1)**: Y0 Q2 deliverable = **R1 Research Report**
  (8-15 page critical review + architecture blueprint).
- **Backup**: Workshop submission (Option C) if empirical work is
  ready by Q2 end.
- **ADR**: decisions/0009-y0q2-deliverable-r1.md
- **Status**: **DECIDED 2026-07-25** (user confirmed)

### Amendment 3: Kill Switch tightened
- **Old (v1.0)**: 6 months no output -> Kill Switch
- **New (v1.1)**: Quarterly check. Failure condition: zero code
  commits + zero draft progress + zero public footprint in one quarter.
  Trigger: pause + "pivot or pause" review.
- **ADR**: decisions/0010-killswitch-quarterly.md
- **Status**: ADOPTED 2026-07-25 (auto-effective)

### Amendment 4: Paper claim made explicit
- **Old (v1.0)**: Project A paper outline stated but novel claim
  underspecified.
- **New (v1.1)**: H1 (decoupling helps, p<0.01 across 16 Procgen games,
  falsifier at delta<0.05 on 12+ games) + H2 (transfer, decoupled
  cross-env AUROC > joint cross-env + 0.1 on 12+ games).
  - Located in `projects/project_a_self_improvement/paper_outline.md`
    Section 0 (Falsifiable Hypotheses).
- **Status**: COMMITTED in paper_outline.md, awaiting paper v0 draft.

### Amendment 5: Critique partner acquisition playbook
- **Old (v1.0)**: "critique partners >= 2 by M6" with no mechanism.
- **New (v1.1)**: Tactical playbook at
  `community/finding_critique_partners.md`. Numbers per channel are
  explicit. Reciprocity is required.
- **Status**: ADOPTED 2026-07-25.

---

## Outstanding amendments queued (v1.2 candidates)

- DEC-007 (Project E P2 -> P1 re-rate): pending user decision
- DEC-006 (grant pitch refresh): pending user decision
- Section 5 collaboration mechanism (still under-specified)

These do not require v1.2 changes; will be folded into v1.2 once
decided.

## v1.1 -> v1.2 (2026-07-25, second-day session)

User asked to "È«×ö" (do everything). This drives further amendments.

### Amendment 6: Project E promoted to P1 (DEC-0007)
- **Old (v1.1)**: Project E at P2 (open question)
- **New (v1.2)**: Project E at P1. Implementation deferred to Dec-Feb
  after Procgen baseline is solid; documentation references from now on.
- **ADR**: decisions/0007-project-e-promote-p1.md
- **Why**: Pearl L3 + AlphaProof + LM/WM integration makes Project E the
  natural third P1 alongside A and C. Defaulted because user punted 3x.

### Amendment 7: Phase 1 first run shipped (2026-07-25)
- **Result**: phase1_20260725_100247.json, 4 games * 1 seed * 50K steps
  in 311s. Mean returns all low (early PPO) confirming we need >= 250K
  per game for Phase 2 (Monitor).
- **Archived**: experiments_log/2026-07-25-phase1-step1-smoke.md

### Amendment 8: Project C paper outline v0 shipped (2026-07-25)
- 6811 bytes in `projects/project_c_causal_world/paper_outline_v0.md`
- H1/H2 hypotheses specified in Section 0 (Slot-WM interventions).
- **Status**: outline only; full paper draft is Y1 deliverable.

## v1.2 -> v1.3 (2026-07-25, AGI continuation turn)

User requested refocus on the AGI program.

### Amendment 9: Project A paper v1_full body shipped
- 11700+ bytes in projects/project_a_self_improvement/paper_outline_v1_full.md
- Title + Abstract + 7-section body + Acknowledgements + References
- Section 4.5 now contains real Phase 1+2 numerical results on
  Procgen coinrun seed 0 (50K PPO steps + 100 train + 50 eval episodes).
- Honest null result documented: Pipeline works, but Phase 1
  too short for Monitor to detect signal.
- Submission target: ICLR 2027 Workshop on Self-Improving Systems.
