# Self-Evaluation Protocol for Codex Sessions

> Established: 2026-07-27 (late evening, after reading F:\test\...\agent-knowledge\)
> Inspired by: KB self-evolving Agent report §2.7 (Honest Boundary) + §2.7 (Self-Review)

## Purpose

Codex operates in single sessions with no persistent memory. To implement
**AIKR mode** (Assumption of Insufficient Knowledge and Resources) honestly,
every session must self-evaluate its work before declaring it "done".

This protocol defines a **5-dimensional self-evaluation** that each session
should run at the end of significant work.

## The 5 Dimensions

For each commit / experiment / paper section, score yourself 1-5 on each:

### 1. Accuracy (1-5)
- Are the numbers correct?
- Are the citations accurate?
- Is the methodology sound?

**Score guide**:
- 5: Verified by independent re-run or formal proof
- 4: Cross-checked with at least one other source
- 3: Plausible based on single-run evidence
- 2: Speculative; needs verification
- 1: Made up or fabricated

### 2. Completeness (1-5)
- Are all TODOs addressed?
- Are negative results reported with same precision as positive?
- Are failure modes acknowledged?

**Score guide**:
- 5: All TODOs done; failure modes documented; pre-registered endpoints met
- 4: Most done; minor TODOs left
- 3: Core done; major gaps acknowledged
- 2: Significant gaps
- 1: Incomplete

### 3. Clarity (1-5)
- Can a smart non-expert understand what was done?
- Are figures / tables clear?
- Is the structure logical?

**Score guide**:
- 5: Clear narrative + visuals + examples
- 4: Clear text but no visuals
- 3: Mostly clear; some sections confusing
- 2: Significant clarity issues
- 1: Incoherent

### 4. Actionability (1-5)
- Can someone reproduce this?
- Are next steps clear?
- Are limitations explicit?

**Score guide**:
- 5: Full reproducibility (seeds, code, data); explicit next steps
- 4: Code present; some setup unclear
- 3: Direction clear; details missing
- 2: Vague direction
- 1: No actionable plan

### 5. Conciseness (1-5)
- Is the writing tight?
- Are unnecessary sections cut?
- Is the essential information prioritized?

**Score guide**:
- 5: Every sentence earns its place
- 4: Mostly tight; minor cruft
- 3: Some unnecessary content
- 2: Verbose
- 1: Bloated

## How to Use

After each significant commit, append to `experiments_log/_self_eval.md`:

```markdown
## Commit <hash>: <title>

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Accuracy | 4/5 | Cross-checked with X but Y needs independent run |
| Completeness | 3/5 | Core done; TODOs: Z |
| Clarity | 5/5 | Clear narrative |
| Actionability | 4/5 | Code present; next steps documented |
| Conciseness | 5/5 | Tight |
| **Overall** | **21/25 (84%)** | Strong contribution |

**Honest Boundary** (this session's unknown unknowns):
- [list what I might be wrong about]
- [list what I did not verify]

**Next session prompt** (for future Codex):
- [what to continue]
- [what to verify]
```

## When to Run Self-Evaluation

- After each significant commit (especially "STRONG POSITIVE" claims)
- After each paper draft update
- After each phase change (Y0 → Y1)
- Before any public announcement

## Why This Matters

Without self-evaluation:
- Fabricated numbers slip through
- Negative results get buried
- Cumulative work degrades in quality

With self-evaluation:
- Each commit has a quality score
- Future sessions can prioritize low-scoring work
- AIKR mode becomes operational, not aspirational

## First Self-Evaluation (2026-07-27)

To be added below as sessions progress.


---

## First Self-Evaluation (2026-07-27 late evening)

After reading F:\test\...\agent-knowledge\ (Agent OS KB), reflecting on my
limitations as single-session Codex agent, and implementing Y0 → Y1 transition.

### Session-level self-evaluation

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **Accuracy** | 4/5 | H1 5/5 verified, Y1.3 +50 verified, AIE recurrent 4x budget confirmed negative. DLR fix verified 95.5%. New: H1 CartPole preliminary AUROC 0.407 may not be statistically meaningful (only seed 0, quick budget). |
| **Completeness** | 4/5 | Y0 main deliverables done. Y1 started: H1 CartPole preliminary. Self-eval protocol just established. Pending: DLR cross-env, MountainCar, paper outline. |
| **Clarity** | 4/5 | README rewritten; thesis v1.0 + addenda; PDFs + HTML. PhD templates ready. |
| **Actionability** | 4/5 | Next steps clear: H1 v2 in background; need 5-seed sweeps. |
| **Conciseness** | 4/5 | Some sessions too verbose. |
| **Overall** | **20/25 (80%)** | Strong Y0 close. |

### Honest Boundary (this session's unknown unknowns)

1. **Y1.3 generalization** is unknown — only tested on LunarLander. The
   claim "Monitor as training-time regularizer" is env-specific.
2. **H1 CartPole preliminary 0.407** may be wrong (only seed 0, only 50 episodes).
3. **AIE 4x budget** showed improvement (-135 vs -127) but not statistically tested.
4. **DLR attention fix** 95.5% is on test set from same distribution; generalization unknown.
5. **No peer review** — all my numbers are self-validated.

### Next session prompt

For the next Codex session that opens `E:\agi-research\`:

1. Check `experiments_log/_h1_v2_cartpole_*.log` — did the v2 run finish?
   If yes, parse result and update H1 cross-env status.
2. Run H1 cross-env on **MountainCar-v0** (sparse reward, like LunarLander) — 5 seeds.
3. Run DLR cross-env on **CartPole-v1** with the 7 predicates.
4. If both MountainCar H1 and CartPole DLR work, write Y1 paper outline.
5. Run self-evaluation per protocol.

### What this session did NOT do (out of scope)

- Find critique partners
- Push to GitHub (already done)
- Apply PhD programs
- Submit NeurIPS paper

These are user actions, not Codex actions.
