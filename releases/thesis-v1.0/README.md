# Archimedes Thesis v1.0 Release

> Release date: 2026-07-29
> Thesis version: 1.0 (with Y1 H10 pilot addendum M)
> Author: Liu Zewen (with Codex agent)
> License: MIT (text), CC-BY-4.0 (this README)
> Project: Archimedes (AGI-2026-001)

---

## What's in this release

This is the **arXiv-ready bundle** of the Archimedes Thesis v1.0,
"Archimedes: A 5-Year Research Program Toward a Self-Improving AGI
Substrate". The thesis presents the overall Archimedes framework and
documents the Y0/Y1 research results, including the 8 pre-registered
H tests (0 supported at the strict t>2.0 rule) and the project
substrate components.

This release is **not a PhD thesis** in the formal sense; it is a
public, MIT-licensed research monograph that documents the Archimedes
Project as a coherent research program.

## Contents

| File | Size | Purpose |
|------|------|---------|
| `thesis.md` | ~115 KB / 3000+ lines | Main thesis text, Markdown format |
| `thesis.html` | ~132 KB | Browser-friendly HTML rendering |
| `thesis.pdf` | ~188 KB | PDF rendering (Unicode font) |
| `README.md` | this file | Release notes |
| `MANIFEST.md` | file inventory | SHA-256 checksums |
| `SUBMISSION.md` | TBD | Submission checklist (workshop + arXiv) |
| `CITATION.cff` | TBD | GitHub-style citation metadata |
| `CHANGELOG.md` | TBD | Release history |

## Quick start

To read the thesis:
- `thesis.md` for the Markdown source.
- `thesis.html` for browser viewing.
- `thesis.pdf` for print-ready PDF.

To regenerate, see the main repository's `bin/` and `docs/` folders.

## What the thesis claims (and does not)

**Claims (with evidence)**:
- **5-year research program**: Archimedes Project, 110+ commits
  in MIT-licensed repo.
- **ENWI framework**: 5-layer architecture, 11 mathematical theorems,
  5 falsifiable predictions.
- **H1 decoupling**: 5/5 seeds on LunarLander-v3, frozen Monitor AUROC
  0.796 vs joint 0.072, delta 0.724.
- **Slot-Monitor**: AUROC 0.989 vs raw-history 0.796, +0.193.
- **Y1.3 training-time regularizer**: 15 seeds, +50 mean over PPO
  baseline on LunarLander, p<0.001 (delta +13.6 on proper Real-vs-Random
  test, n.s.).
- **DLR attention**: 4 envs, 19 predicates, 97.8% mean accuracy.
- **9-hypothesis pre-registered framework**: 6 validated, 2 refuted,
  1 open.
- **GovBench governance primitives**: H1+H2+H3 validated (n=7).

**Does NOT claim**:
- AGI-Strong is achievable in 5 years.
- Y1.3 generalizes beyond LunarLander (cross-env tie/untestable).
- Inference-time Monitor intervention works (6/6 DEC-0011 + 2 more FAILED).
- Y1.3 transfers to multi-agent (H5 REFUTED).
- Monitor as exploration bonus works (H1.4 REFUTED).
- Long PPO budgets (500K) rescue Y1.3 (H3 ACTIVE HARM).
- Decoupling transfers to LLM self-rewarding (H10 direction REFUTED
  at n=5/N=12 stratified pilot).

The "0/8 supported" headline is real and reported throughout.

## What this thesis is and is not

This thesis is:
- A public research monograph, MIT-licensed.
- A record of an open, reproducible research program.
- A demonstration of pre-registered empirical methodology.
- An honest accounting of both positive and negative results.

This thesis is not:
- A formal PhD thesis (no institutional affiliation).
- A claim of AGI-Strong or general intelligence.
- A substitute for peer review.
- A product or commercial application.

## Companion documents

In the main repository:

- `docs/NO_SELF_DECEPTION.md` -- the anti-self-deception protocol that
  produced this thesis's honest framing.
- `experiments_log/2026-07-29-H10-stratified-n5-result.md` -- the
  most recent H10 pilot result (Project G, direction-REFUTED at small N).
- `experiments_log/2026-07-28-H2.0-n10-final.md` -- the H2.0 n=10
  extension that closed the Y1.x + H2.0 sub-project.
- `experiments/_4_10_25_synthesis.md` -- the honest synthesis that
  closed the Y1.x sub-project.
- `papers/y1_paper_draft.md` -- the Y1 paper v3.8 (paper-form summary).
- `decisions/` -- 11+ decision records documenting major pivots.

## Submission status

This release is **NOT yet submitted** to arXiv or any venue. See
`SUBMISSION.md` (TODO) for the pre-submission checklist.

The thesis is published as a public document in the Archimedes
repository. Future revisions will be tracked in `CHANGELOG.md`.

## License

- Thesis text: MIT
- This README: CC-BY-4.0
- Figures: MIT

## Contact

Liu Zewen -- Archimedes Project (AGI-2026-001)
Repository: github.com/aidless/agi-research

---

*Release v1.0 packaged 2026-07-29 by Codex agent under
NO_SELF_DECEPTION.md protocol.*