# Thesis v3.0 PDF Build Status (2026-08-01)

**Status:** v3.0 PDF generated (= v2.0 PDF, 470 KB, 72 pages)
**Note:** Full v3.0 PDF (Parts I-X) is BLOCKED on a v2.0 source LaTeX issue
discovered during integration. The Y5 v1.3 master synthesis is appended
as a separate PDF (`papers/y5_v1_3_master_synthesis.pdf`, 1.37 MB, 89 pages).

## What's in the v3.0 PDF (= v2.0 PDF)

The v3.0 PDF currently equals the v2.0 PDF (470 KB, 72 pages) and
contains:

- Title, abstract, table of contents
- Part I: Foundations
- Part II: Project A -- Self-Improvement via Decoupled Monitors
- Part III: Project C -- Causal World Models with Slot Attention
- Part IV: Project D -- Language Interface
- Part V: Project E -- Neuro-Symbolic Verification
- Part VI: Project F -- Multi-Agent (Sketch)
- Part VII: Cross-Environment and Transfer
- Part VIII: Discussion and Future Work
- Part IX: Project G -- LLM Self-Monitoring (Y4)
- Appendices
- References
- Final Notes
- Addenda

## What's NOT in the v3.0 PDF (deferred)

- Part X: Y5 v1.3 master synthesis (the headline 89-page result)
  - Available separately at papers/y5_v1_3_master_synthesis.pdf

## Why v3.0 PDF is not a single combined file

The thesis_draft_v3.0.tex source (338 KB, 8794 lines) attempts to combine
v2.0 body with Y5 as Part X. The build fails at line 222 with
"Misplaced \\noalign" in a tabular environment. The v2.0.tex source
builds cleanly on its own (verified 2026-08-01 00:30). The issue is in
the wrapper that joins v2.0 with Y5.

The fix is one of:
1. Repair the v2.0 body tabular at line 222 (specifically the Path 1-5
   table that begins around line 219) to use booktabs-compliant \\hline
   position
2. Use a different wrapper (e.g., \\input{v2.0_body} + \\input{y5_body}
   instead of inlining the body)
3. Render v2.0 and Y5 separately and merge with pdfunite

## Proposed fix (option 3, easiest)

```bash
# After v2.0.tex and Y5 v1.3.tex are both compiled:
pdfunite thesis_draft_v2.0.pdf papers/y5_v1_3_master_synthesis.pdf thesis_draft_v3.0.pdf
```

This produces a single 470KB + 1370KB = ~1840 KB / 161-page PDF containing
all 10 thesis parts.

## Build verification (v2.0 only)

```
$ pdflatex thesis_draft_v2.0.tex
... (build log) ...
Output written on thesis_draft_v2.0.pdf (72 pages, 470568 bytes).
```

The v2.0 PDF is the thesis v3.0 baseline. Y5 v1.3 PDF is the
master-synthesis companion. Together they form the v3.0 thesis content.

## Files for v3.0 bundle

```
thesis_draft_v3.0.pdf           470 KB   <- v2.0 PDF (Parts I-IX + Addenda)
papers/y5_v1_3_master_synthesis.pdf  1.37 MB  <- Y5 v1.3 master synthesis (Part X)
thesis_draft_v3.0.tex          338 KB   <- Combined .tex source (not buildable)
thesis_v3_0_overview.md       8.7 KB   <- Markdown overview of v3.0 structure
thesis_v3_0_abstract.md       1.6 KB   <- Markdown extract of v2.0 abstract
```

For the COLM 2026 submission, only the Y5 v1.3 master synthesis PDF
is uploaded (in arxiv_submission.tar.gz). The thesis v3.0 is a
broader research-direction document, not a separate submission.

## Future work (post-P3)

After the P3 hybrid pre-reg completes (2026-08-01 ~01:30):
1. Re-build thesis_draft_v3.0.tex with the Y5 v1.3.1 content (P3 result
   incorporated)
2. Attempt pdfunite merge of v2.0 + v1.3.1 (if v3.0.tex wrapper
   still has issues)
3. Generate a single combined v3.0.1 PDF for archival purposes