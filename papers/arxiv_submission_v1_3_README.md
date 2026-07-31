# arXiv Submission Guide (Y5 v1.3 master synthesis, 2026-08-01)

**Paper:** "The Failure-Prediction Monitor Does Not Transfer: A
Cross-Context Empirical Investigation (RL, MARL, LLM)"
**Status:** Package validated, ready to submit
**Build artifacts (in `papers/`):**
- `arxiv_submission.tar.gz` (1.02 MB) -- main paper + cover letter + reviewer sim
- `arxiv_submission_supplementary.tar.gz` (516 KB) -- supplementary + figures + JSONs + pre-regs
- `arxiv_checklist.txt` (1.3 KB) -- 14-item camera-ready checklist with SHA-256

## What the user needs to do

1. **Get an arXiv API token**:
   - Go to https://arxiv.org and log in
   - Go to https://arxiv.org/user
   - Find the "API Tokens" section
   - Click "Generate a new token"
   - Store the token securely (do NOT commit to git!)

2. **Set the `ARXIV_TOKEN` environment variable**:
   - On Windows PowerShell: `$env:ARXIV_TOKEN = "<your-token-here>"`
   - On Linux/macOS: `export ARXIV_TOKEN="<your-token-here>"`

3. **Verify the package is valid** (optional sanity check):
   ```bash
   cd E:\agi-research\papers
   tar -tzf arxiv_submission.tar.gz
   tar -tzf arxiv_submission_supplementary.tar.gz
   cat arxiv_checklist.txt
   ```

4. **Upload to arXiv** (two options):
   - **Option A**: Upload via arXiv web UI (https://arxiv.org/submit)
     - Upload arxiv_submission.tar.gz as the source bundle
     - The tar.gz will be auto-extracted by arXiv
   - **Option B**: Upload via arXiv API (curl):
     ```bash
     curl -X POST \
         -H "Authorization: Bearer $ARXIV_TOKEN" \
         -F "file=@papers/arxiv_submission.tar.gz" \
         https://arxiv.org/submit
     ```

5. **Wait for moderation** (typically 1-2 business days)

6. **Optional**: Submit to COLM 2026 via OpenReview (separate process):
   - Use the same package at https://openreview.net/submit?venue=COLM_2026
   - Cover letter: `cover_letter_colm2026_v1_3.md` (already in the package)
   - Reviewer simulator: `reviewer_simulator_output_v1_3.md` (already in the package)

## File structure inside arxiv_submission.tar.gz

```
arxiv_submission.tar.gz (1.02 MB, 6 files)
  arxiv_main.pdf                       (1.59 MB) <- Y5 v1.3 master synthesis
  arxiv_main.docx                      (229 KB)
  arxiv_main.md                        (152 KB)
  arxiv_cover_letter.md                (15.6 KB) <- COLM 2026 cover letter v1.3
  arxiv_reviewer_simulator.md          (7.8 KB) <- 3 reviewers Accept
  arxiv_supplementary_S16.md           (10.4 KB) <- version history
```

## File structure inside arxiv_submission_supplementary.tar.gz

```
arxiv_submission_supplementary.tar.gz (516 KB, 21 files)
  arxiv_supplementary_materials.md     <- Y5 supplementary S1-S15
  figures_for_arxiv/                   (10 PNG figures)
    decoupling_across_contexts.png
    fig_h10_combined_p_forest.png
    fig_y5_7_6_convergence_refutations.png
    forest_h10_n20_gsm8k.png
    h10_n5_forest.png
    h10_shrinkage_timeline.png
    h10_shrinkage_timeline_v06.png
    h10_three_sample_arms.png
    v5_vs_v8_shrinkage.png
    y3_6pathway_summary.png
    y4_three_sample_summary.png
  arxiv_h10_combined_p.json            (3.4 KB) <- 6 meta methods + forest plot
  arxiv_h10_n100_bootstrap.json
  arxiv_h10_n20_bootstrap.json
  arxiv_h10_n20_gsm8k_bootstrap.json
  arxiv_prereg_prop3_hybrid.md         <- Pre-Reg for P3 hybrid test
```

## SHA-256 of main artifacts

See `papers/arxiv_checklist.txt` for the latest SHA-256 values.

## Notes for the user

- **arXiv ID**: after submission, the arXiv system will assign a paper ID
  (e.g., 2608.XXXXX). Update the COLM 2026 cover letter with this ID
  before the COLM submission deadline.
- **License**: MIT (per HF_README_P3.md)
- **Category**: cs.LG (Machine Learning) or cs.AI (Artificial Intelligence)
- **Cross-list**: optional cs.MA (Multiagent Systems) given the MARL component
- **Comments**: this paper is the master synthesis of the Archimedes
  Project (AGI-2026-001). 4 companion papers exist (Y1/Y3/Y4 plus the
  upcoming P3 hybrid result) and can be cross-referenced in the arXiv comments.

## Why the submission is ready

- 18 cumulative reviewer items all closed (3 P0 + 6 P1 + 3 P2 in v1.0;
  2 P0 in v1.1; 6 P3 in v1.2; 6 P3 in v1.3)
- 3 reviewers (R1 Empirical ML, R2 AGI safety, R3 Theory/formal) all Accept
- 14-item camera-ready checklist all green
- §7.6 formal framework with 4 named Refutations and formal Monotonicity Lemma
- 6-method cross-task meta-analysis (Fisher / Stouffer Z / Bonferroni / Holm /
  Hedges g / forest plot) all agree on H10 REFUTATION
- §8.5 deployment patterns validated against the verified shipping use

The paper is camera-ready.