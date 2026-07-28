# Y1 Paper Release v3.7 -- Decoupled Monitors as Training-Time Regularizers

> Release date: 2026-07-28
> Paper version: 3.7 (final pre-arXiv draft)
> Author: Liu Zewen (with Codex agent)
> License: MIT (code) / CC-BY-4.0 (paper text)
> Project: Archimedes (AGI-2026-001)

---

## What's in this release

This is the **arXiv-ready bundle** of the Y1 paper, "Decoupled Monitors
as Training-Time Regularizers", together with its figures, tables,
9-hypothesis framework, and related-work notes.

The paper reports both the headline result (Y1.3 +50 mean over PPO
baseline on LunarLander-v3, n=15 seeds, p<0.001) and the broader null
synthesis (8 pre-registered H tests, 0 supported at the pre-registered
decision rule t > 2.0; honest cross-env and inference-time negative
results). It is written under the NO_SELF_DECEPTION.md protocol
(`docs/NO_SELF_DECEPTION.md`).

## Contents

| File | Size | Purpose |
|------|------|---------|
| `paper.md` | ~38 KB / 987 lines / ~5600 words | Main paper text, Markdown format |
| `figures/y1_fig1_y13_lunarlander.png` | ~30 KB | Figure 1: Y1.3 vs PPO box plot (LunarLander) |
| `figures/y1_fig2_y13_per_seed.png` | ~36 KB | Figure 2: per-seed scatter (15 seeds) |
| `figures/y1_fig3_dlr_crossenv.png` | ~45 KB | Figure 3: DLR cross-env (4 envs, 19 predicates) |
| `figures/y1_fig4_y13_lambda.png` | ~45 KB | Figure 4: Y1.3 lambda sweep (6 values) |
| `tables/y1_table1_dlr_summary.tex` | ~0.8 KB | Table 1: DLR per-env summary (LaTeX) |
| `tables/y1_table2_y13_summary.tex` | ~0.8 KB | Table 2: Y1.3 per-seed (LaTeX) |
| `y1_9hypothesis_framework.md` | ~12 KB | Supplementary: 9-hypo framework (6 valid, 2 refuted, 1 open) |
| `related_work_4systems.md` | ~13 KB | Supplementary: 4-systems related work |
| `README.md` | this file | Release notes |
| `MANIFEST.md` | TBD | File inventory with SHA-256 |
| `SUBMISSION.md` | TBD | arXiv submission checklist |
| `CITATION.cff` | TBD | GitHub-style citation metadata |
| `CHANGELOG.md` | TBD | Release history |

## Quick start

To read the paper, open `paper.md` in any Markdown viewer (VS Code,
Typora, GitHub). The paper is single-column, ~14 pages when rendered
to PDF (estimated from word count).

To regenerate figures, see `../papers/make_figures.py` in the main
repository (requires Python 3.10+, PyTorch 2.0+, numpy 1.24+).

To reproduce results, see Appendix D in `paper.md` and the main
repository README.

## What this paper claims (and what it does not)

**Claims (with evidence)**:
- **H1 decoupling** (5/5 seeds, LunarLander-v3, AUROC frozen 0.796 vs
  joint 0.072, delta 0.724). VALIDATED.
- **Y1.3 Monitor-as-reward-shaper** (15 seeds, LunarLander-v3, +50
  mean over PPO baseline, p<0.001). VALIDATED on LunarLander.
- **DLR attention** (4 envs, 19 predicates, 97.8% mean accuracy).
  VALIDATED cross-env.
- **9-hypothesis pre-registered framework** (6 validated, 2 refuted,
  1 open). Methodological contribution.

**Does NOT claim**:
- That Y1.3 generalises beyond LunarLander (cross-env: tie on Acrobot,
  undefined on MountainCar where PPO fails to converge).
- That inference-time Monitor intervention works (6/6 DEC-0011
  variants FAILED + 2 more failures).
- That Monitor as exploration bonus works (H1.4 REFUTED, 1/5 positive,
  delta -25.6).
- That Monitor as PPO value baseline works (Y1.4 NOT supported).
- That 500K PPO budget rescues Y1.3 (H3: -53.1 vs random signal,
  ACTIVE HARM at long training).

The "8 pre-reg tests, 0 supported" headline is real and is reported
in the paper's Section 4.10.25-4.10.27.

## What was deliberately NOT included in this release

- The Y2 Phase 2 multi-agent results (still in progress; DMC H5
  REFUTED on continuous, MADDPG v2 baseline working +7.7 vs random).
  These belong in a separate Phase 2 paper.
- The H6 instrumented mechanism test (Spearman rho mean +0.14, 3/5
  seeds REFUTED monotonic-decrease claim). This is mentioned in the
  paper's discussion but not given a full section.
- Co-author / lab affiliations (single-author work; PI may add
  affiliations before submission).
- The thesis v1.0 (separate artifact; ~103 KB; available at
  `../thesis_draft_v1.0.md` in the main repository).

## Companion documents

In the main repository:

- `docs/NO_SELF_DECEPTION.md` -- the protocol that produced this
  paper's honest framing.
- `experiments_log/2026-07-28-H2.0-n10-final.md` -- the H2.0 n=10
  extension that confirmed the 8/8 null result.
- `experiments/_4_10_25_synthesis.md` -- the Section 4.10.25 honest
  synthesis that closes the Y1.x sub-project.
- `experiments_log/2026-07-28-h6-instrumented-5seed.md` -- the H6
  mechanism test that was REFUTED.
- `decisions/` -- the 11 decision records (DEC-0001 .. DEC-0011+)
  that document the agent's self-correction sequence.

## Submission status

This release is **NOT yet submitted** to arXiv. See `SUBMISSION.md`
for the pre-submission checklist.

Estimated submission window: late 2026 (after PI review and any
final revisions).

## License

- Paper text: CC-BY-4.0
- Code in companion repository: MIT
- Figures: CC-BY-4.0
- Tables: CC-BY-4.0

## Contact

Liu Zewen -- Archimedes Project (AGI-2026-001)
Repository: github.com/aidless/agi-research

---

*Release v3.7 packaged 2026-07-28 by Codex agent under
NO_SELF_DECEPTION.md protocol.*