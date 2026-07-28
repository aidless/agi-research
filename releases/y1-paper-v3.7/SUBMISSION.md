# arXiv Submission Checklist -- Y1 Paper v3.7

> Date: 2026-07-28
> Target venue: arXiv (cs.LG / cs.AI)
> Submission window: late 2026 (after PI final review)
> This checklist is for the PI to walk through before clicking Submit.

---

## 1. Pre-submission checks (PI must verify)

- [ ] **Single-author affiliation block**: confirm affiliation, ORCID,
      email. Current draft is anonymous (Archimedes / AGI-2026-001
      project header). PI must add real affiliation if submitting
      under personal name.

- [ ] **Title**: "Decoupled Monitors as Training-Time Regularizers:
      An Empirical Study with 9 Pre-Registered Hypotheses". This is
      a working title; PI may revise.

- [ ] **Abstract**: 6 paragraphs in current draft. arXiv limit is
      ~1920 chars (~300 words). Current abstract is ~430 words;
      may need to be trimmed. PI to verify.

- [ ] **MSC/ACM classifications**: cs.LG (primary), cs.AI, cs.MA
      (secondary). arXiv submission form needs these.

- [ ] **License selection on arXiv**: CC-BY-4.0 (paper text). Code
      is MIT in the companion repository; not submitted to arXiv.

- [ ] **Conflict of interest disclosure**: none (single-author
      independent work; no funding source declared). PI to add
      if any funding exists.

- [ ] **No simultaneous submission**: confirm not under review at
      a journal/conference. (Y1 paper is currently not submitted
      anywhere; safe to submit to arXiv first.)

## 2. File preparation (PI must do)

arXiv accepts a single .tar.gz or .zip with the source files. The
submission should include:

- [ ] `paper.md` (Markdown) **OR** `paper.tex` (LaTeX) -- Markdown
      will be auto-converted by arXiv; LaTeX is preferred for
      better typesetting. PI to choose.

- [ ] All 4 figures in `figures/` (PNG, embedded in Markdown or
      referenced from LaTeX).

- [ ] Both tables in `tables/` (LaTeX source). PI to choose whether
      to inline in main text or appendix.

- [ ] `y1_9hypothesis_framework.md` and `related_work_4systems.md`:
      these are supplementary. Decide whether to include in the
      main tarball or upload as ancillary files.

- [ ] `biblio.bib` (BibTeX) if converting to LaTeX. The paper's
      references are inline in Markdown currently.

- [ ] `00README.XXX` (arXiv convention): a brief README that
      describes the bundle contents. The current `README.md` works.

## 3. PDF generation options

If PI wants a PDF for arXiv (arXiv accepts both source + PDF, but
the PDF must be generated from the source):

- [ ] **Option A**: pandoc
  `pandoc paper.md -o paper.pdf --bibliography=biblio.bib --citeproc`
  (requires LaTeX install + pandoc-citeproc)

- [ ] **Option B**: manuscript-tools / quarto
  `quarto render paper.md --to pdf`

- [ ] **Option C**: arXiv auto-converts the Markdown source.
  Less control over formatting but easier.

## 4. What is NOT being submitted

- The full thesis (~103 KB, 3000+ lines) is **not** part of this
  arXiv submission. It lives in the companion repository.
- The Phase 2 multi-agent work (DMC, MADDPG v2) is **not** part of
  this submission. It is a separate paper (see
  `papers/phase2_paper_outline.md`).
- The GovBench governance results (H7, H8) are **not** part of the
  main paper but could be added as supplementary. PI to decide.
- The full 9-hypothesis framework is in the main paper but is also
  available as a standalone document in this bundle.

## 5. Final quality gates

Before clicking submit, the PI should confirm:

- [ ] **Every positive claim has a negative control** (or a stated
      reason why no control was run). The paper does this in
      Section 4.10.25.
- [ ] **Per-seed numbers are reported, not just aggregates**.
      The paper does this in Appendix A.1 (Y1.3 15-seed) and
      Section 4.10.25.
- [ ] **Pre-registration is documented**. The paper does this
      in Section 4.10.26 and references the 8 pre-registered
      H tests.
- [ ] **NO_SELF_DECEPTION.md compliance**: the paper reports
      the v1.0 -> v1.1 -> v1.2 -> v1.3 -> v1.4 self-correction
      sequence, not just the final verdict. This is in
      Section 4.10.25.3.
- [ ] **Limitations are explicit**. Section 6 does this. Key
      limitations: single-env headline result, no peer review,
      single-author independent work, no replication by an
      external lab.

## 6. After submission

Once the paper is on arXiv:

- [ ] Update `releases/y1-paper-v3.7/CHANGELOG.md` with the
      arXiv ID and submission date.
- [ ] Update the README in the companion repository to link
      to the arXiv version.
- [ ] Post the canonical link (not just the PDF) to Twitter /
      Discord / Email drafts (in `community/`).
- [ ] Add the arXiv preprint citation to the SoP
      (`phd_applications/programs/*/statement_of_purpose.md`).

## 7. Risk assessment

What could go wrong:

- **Single-author attribution**: arXiv may require a real
  institutional affiliation. PI to verify.
- **Self-correction disclosure**: the paper's "publisher's
  paradox" framing (Section 4.10.25.3) is unusual; some reviewers
  may misread it as overclaim. This is intentional; the NO_SELF_
  DECEPTION protocol makes the disclosure explicit.
- **No external replication**: until another lab replicates the
  Y1.3 +50 result, the headline is single-source. arXiv is OK
  with this (preprint server) but downstream publication will
  require independent replication.
- **Pre-registration timeline**: the paper discloses in Appendix
  D.6 that the experiments were NOT pre-registered in the
  traditional sense. This is honest; arXiv does not require
  pre-registration.

## 8. Estimated time

- Single-author affiliation + title/abstract polish: 1-2 hours.
- Markdown -> LaTeX conversion (if chosen): 3-4 hours.
- tar.gz packaging: 15 minutes.
- arXiv submission form: 30 minutes.

Total PI time: ~5-7 hours for a clean submission.

---

*Checklist prepared 2026-07-28 by Codex agent. PI to walk through
each item before submission.*