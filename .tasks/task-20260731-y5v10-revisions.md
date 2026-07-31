# Task: Y5 v1.0 Revision -- 12 Minor Items from Reviewer Simulator

**Generated:** 2026-07-31
**Owner:** Codex (under Liu Zewen's supervision)
**Source:** `papers/reviewer_simulator_output_v1_0.md` (3 reviewers, all Weak Accept)
**Target paper:** `papers/y5_v1_0_master_synthesis.{md,pdf}` (Y5 master synthesis v1.0)
**Estimated cost:** 2-3 weeks focused work, **no new experiments required**

## Goal

Address all 12 required minor revisions identified by the reviewer simulator and produce a v1.1 camera-ready draft suitable for COLM 2026 submission.

## Priority legend

- **P0** -- blocking for submission; required
- **P1** -- strongly recommended; required for camera-ready
- **P2** -- nice-to-have; improves clarity but not blocking

## Sub-tasks (ordered by reviewer)

### R1 (Empirical ML researcher) -- 4 items

- [ ] **R1.1 (P1)** Add 1-paragraph justification for the 3-convergence-conditions decomposition vs alternatives.
  - **Action:** Insert paragraph after Definition 7 in `y5_monitor_transfer_synthesis.md` §7.6.1 arguing that (i) the 3 conditions are mutually exclusive failure modes, (ii) each is observable in the empirical record, (iii) collapsing any two would lose predictive specificity.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md`
  - **Effort:** 0.5 day

- [ ] **R1.2 (P1)** Add Pre-Reg plan + sample-size estimate for Proposition 3 (hybrid > either alone).
  - **Action:** Write `experiments_log/2026-08-XX-PRE-REG-P3-hybrid.md` with the proposed test, expected effect size, and required n for 80% power. Reference it from Proposition 3.
  - **Files:** `experiments_log/2026-08-XX-PRE-REG-P3-hybrid.md` (new), `papers/y5_monitor_transfer_synthesis.md` (cross-ref)
  - **Effort:** 1 day

- [ ] **R1.3 (P1)** Footnote the assumed effect size in the required-n-for-80%-power calculation in Proposition 2.
  - **Action:** Add footnote after "required n = 723" specifying that this assumes the observed d = +0.030 (Cohen's d at n=100 simple arith).
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.2
  - **Effort:** 0.25 day

- [ ] **R1.4 (P0)** Add combined-p test for cross-task consistency across the 4 H10 sample sizes.
  - **Action:** Compute Fisher combined p across (n=5 arith, n=20 arith, n=100 arith, n=20 GSM8K). Use the existing per-seed JSONs. Insert a row into the §4.4 cross-task table.
  - **Files:** `experiments_log/_h10_combined_p.json` (new), `papers/y5_monitor_transfer_synthesis.md` §4.4
  - **Effort:** 1 day

### R2 (AGI safety researcher) -- 4 items

- [ ] **R2.1 (P1)** Compute-cost estimate for R4 (Monitor at 7B / 70B LLM scale).
  - **Action:** Add a paragraph after Refutation 4 in §7.6.3 estimating GPU-hours (e.g., 7B pilot = 200 GPU-h on a single A100; 70B pilot = 4000 GPU-h). State whether the authors have access.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.3
  - **Effort:** 0.25 day

- [ ] **R2.2 (P1)** 2-3 concrete verification deployment patterns in §8 with their failure modes.
  - **Action:** Add §8.1 (runtime guardrail), §8.2 (DLR predicate in critic), §8.3 (pre-commit review) subsections. Each lists a concrete pattern, an example, and 1-2 known failure modes.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §8
  - **Effort:** 1 day

- [ ] **R2.3 (P2)** First-principles (PAC-learning / distribution-shift) derivation of the 3 Convergence Conditions.
  - **Action:** Add §7.5.5 "First-principles motivation" deriving each condition from a corresponding theorem (PAC-learnability for Condition 3, distribution-shift for Condition 1, information-theoretic for Condition 2).
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.5.5 (new), bibliography update
  - **Effort:** 2 days

- [ ] **R2.4 (P1)** Move framework-presentation limitations (P3 untested, required-n sensitivity, decomposition uniqueness) into §9.
  - **Action:** Add §9.5 "Limitations of the formal framework" listing the 3 framework-presentation limitations as bullet points.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §9.5 (new)
  - **Effort:** 0.5 day

### R3 (Theory / formal methods) -- 4 items

- [ ] **R3.1 (P0)** Make the implicit assumption in Proposition 1's converse direction explicit.
  - **Action:** Add Assumption A1 (positive mutual information between auxiliary signal and policy value function) at the start of §7.6.2. State that without A1, the converse is false.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.2
  - **Effort:** 0.5 day

- [ ] **R3.2 (P2)** Add Hanley-McNeil bound derivation for Definition 6.
  - **Action:** Add §7.6.1 footnote linking Definition 6 (Condition 3) to Hanley & McNeil (1982) "The meaning and use of the area under a ROC curve" for the AUROC standard error bound.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.1, bibliography
  - **Effort:** 0.5 day

- [ ] **R3.3 (P1)** State R1-R4 as a logical disjunction in the formal text.
  - **Action:** Add explicit sentence: "The framework is falsified IFF at least one of R1, R2, R3, R4 is observed."
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.3
  - **Effort:** 0.25 day

- [ ] **R3.4 (P2)** Monotonicity discussion: does observing R1 alone vs R1+R2 imply different framework updates?
  - **Action:** Add §7.6.6 "Monotonicity" with a short proof that R1 alone is consistent with "Monitor in non-stationary contexts needs periodic retraining" while R1+R2 is consistent with "Monitor in LLM contexts fundamentally lacks the right features". These are different updates.
  - **Files:** `papers/y5_monitor_transfer_synthesis.md` §7.6.6 (new)
  - **Effort:** 1 day

## Dependencies

```
R1.4 (combined-p)        standalone
R1.3 (footnote)          standalone
R1.1 (decomposition)     standalone
R1.2 (Pre-Reg P3)        standalone
R2.1 (R4 compute)        standalone
R2.2 (deployment)        standalone
R2.4 (limitations)       standalone
R2.3 (first-principles)  needs R1.1 done first
R3.1 (A1 explicit)       standalone
R3.3 (disjunction)       standalone
R3.2 (Hanley-McNeil)     standalone
R3.4 (monotonicity)      needs R3.3 done first
```

## Risks (top 3)

1. **R1.4 combined-p test**: If Fisher combined p is NOT significant despite 4 REFUTATIONS, the framework's "consistent direction" claim weakens. Mitigation: pre-compute and have a backup narrative.
2. **R2.3 first-principles derivation**: A poor derivation may make the framework look more speculative, not less. Mitigation: keep the first-principles section as a "motivation" not a "proof"; cite established theorems.
3. **R3.1 A1 explicit**: Naming the implicit assumption may invite a reviewer to construct a counterexample where A1 fails. Mitigation: discuss what happens when A1 fails (the signal can still be useful for verification, just not as a training signal).

## Estimated compute

- **CPU-minutes:** ~120 (combined-p computation + numerical footnote + monotonicity sketch)
- **GPU-hours:** 0 (no new experiments)

## Cancellation criterion

Stop the revision if any of:

1. Combined-p test (R1.4) returns p > 0.05 and the framework's cross-task consistency claim has to be retracted.
2. First-principles derivation (R2.3) reveals the 3 Convergence Conditions are not derivable from a single theorem family (would force a framework rewrite).
3. R3.1 explicit assumption (A1) is contradicted by an obvious counterexample (would force a Proposition 1 rewrite).

In any of these cases, mark the task BLOCKED and ask the user for guidance before continuing.

## Sub-task summary

| ID | Title | Priority | Effort | Owner |
|---|---|---|---|---|
| R1.1 | 3-condition decomposition justification | P1 | 0.5 d | TBD |
| R1.2 | Pre-Reg for Proposition 3 | P1 | 1 d | TBD |
| R1.3 | Footnote required-n assumption | P1 | 0.25 d | TBD |
| R1.4 | Combined-p cross-task consistency | P0 | 1 d | TBD |
| R2.1 | Compute cost for R4 | P1 | 0.25 d | TBD |
| R2.2 | 3 deployment patterns in §8 | P1 | 1 d | TBD |
| R2.3 | First-principles derivation | P2 | 2 d | TBD |
| R2.4 | Framework-presentation limitations in §9.5 | P1 | 0.5 d | TBD |
| R3.1 | Make A1 explicit in Proposition 1 | P0 | 0.5 d | TBD |
| R3.2 | Hanley-McNeil bound for Condition 3 | P2 | 0.5 d | TBD |
| R3.3 | R1-R4 logical disjunction | P1 | 0.25 d | TBD |
| R3.4 | Monotonicity discussion | P2 | 1 d | TBD |
| **Total** | | | **~9 person-days** | |

## Hand-off

When all 12 items are checked off:

1. Re-run `E:\gen_pdf.py` to regenerate the v1.1 PDF.
2. Update `E:\ObsidianKnowledgeBase\01 - Papers\Y5 v1.0 Master Synthesis\` to "Y5 v1.1 Master Synthesis".
3. Commit with message pattern "Y5 v1.1: address reviewer simulator items R{1,2,3}.{N}".
4. Re-run the reviewer simulator (3 reviewers) on v1.1 to verify all Weak Accept verdicts hold.