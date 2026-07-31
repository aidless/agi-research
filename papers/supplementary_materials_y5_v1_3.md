# Supplementary Materials (Y5 v1.3 master synthesis)

**Paper:** "The Failure-Prediction Monitor Does Not Transfer: A
Cross-Context Empirical Investigation (RL, MARL, LLM)"
**Authors:** Liu Zewen + Codex (Archimedes Project, AGI-2026-001)
**Date:** 2026-07-31 (Y5 v1.3) / 2026-08-01 (COLM 2026 submission)
**Pre-registration chain:** 4 documents (H10 original + Amendment 1 + Addendum + P3 hybrid)

This supplementary materials document provides the detailed data,
code, and reproducibility instructions for the Y5 v1.3 master
synthesis. For per-paper supplementary materials, see the
companion paper files (`papers/y1_paper_draft.md`,
`papers/monitor_signal_vs_dlr_6pathway.md`,
`papers/project_g_v0_5_h10_paper.md`).

## S1. Repository layout

All code, data, and analysis scripts are in the Archimedes Project
git repository: https://github.com/aidless/agi-research

```
agi-research/
  papers/
    y5_v1_3_master_synthesis.{md,html,docx,pdf}     # Y5 master synthesis (89 pages)
    y5_monitor_transfer_synthesis.tex              # Y5 LaTeX source
    y1_paper_draft.md                              # Y1 v1.0 (28 pages)
    monitor_signal_vs_dlr_6pathway.md              # Y3 v1.0 (17 pages)
    project_g_v0_5_h10_paper.md                    # Y4 v1.0 (PDF)
    y*_v1_*_paper.{html,docx,pdf}                 # Rendered companion papers
    supplementary_materials.md                     # This file
    supplementary_S16_version_history.md          # S16: changelog v0.8 -> v1.3
    cover_letter_colm2026_v1_3.md                  # COLM 2026 cover letter
    reviewer_simulator_output_v1_3.md              # Final reviewer sim
    arxiv_submission.tar.gz + arxiv_submission_supplementary.tar.gz
    arxiv_checklist.txt                            # 14-item camera-ready checklist
    arxiv_submission_v1_3_README.md                # arXiv upload guide
    v1_3_1_P3_changelog_template.md                # P3 result changelog (TBD)
  projects/
    project_a_self_improvement/                    # Y1 code
    project_f_multi_agent/                        # Y3 + P3 code
      code/
        pz_maddpg_v8.py                            # P3 training script (3 arms)
        r1_test.py                                 # R1 test scaffold
    project_g_llm_self_monitoring/                 # Y4 code
  experiments_log/
    # Y1 results
    _pz_maddpg_v5_15seed_*.log
    # Y3 results
    _v8_10k_n50_*.log
    _h10_*_bootstrap.json                          # Y3 v8 dlr_only bootstrap
    # Y4 results (4 H10 sample sizes)
    _h10_n5_perseed.json
    _h10_n20_perseed.json
    _h10_n100_perseed.json
    _h10_n20_gsm8k_perseed.json
    _h10_n20_bootstrap.json
    _h10_n100_bootstrap.json
    _h10_n20_gsm8k_bootstrap.json                  # H10 GSM8K 200-token (kill switch verdict)
    # Y4 pre-reg chain
    2026-07-28-PRE-REGISTERED-H10.md
    2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md
    2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md
    # Y5 v1.1+ cross-task meta-analysis
    _h10_combined_p.json                            # 6 meta-analytic methods
    # Y5 v1.2+ supplementary
    fig_h10_combined_p_forest.png                  # Forest plot visualization
    fig_h10_combined_p_forest_source.html          # SVG build source
    # P3 hybrid (in progress as of 2026-08-01)
    _p3_hybrid_<arm>_s<seed>.log                   # Per-job logs
    _p3_hybrid_bootstrap.json                       # Bootstrap aggregator output
    # P3 hybrid pre-reg
    2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md    # P3 pre-reg
    _run_v8_10k_n50_3arm.ps1                        # Full pre-reg launcher
    _run_p3_hybrid_production.ps1                   # Tightened production launcher
    # R1 test (scaffold)
    r1_test.py                                      # R1 test (3 arms)
    _run_r1_test.ps1                                # R1 launcher
    # Aggregator
    _agg_p3_hybrid.py                               # Bootstrap aggregator
    # Smoke test
    _smoke_p3_*.log                                 # 3-arm smoke test
    _smoke_test_results.md                          # Smoke test documentation
```

## S2. Empirical chain overview

The 11 empirical comparisons in Y5 v1.3 come from 3 sub-projects:

### Y1: Single-agent RL (1 comparison)

- Y1.3 Monitor on LunarLander-v3, n=15 seeds, +39.5 mean improvement
- Source: `projects/project_a_self_improvement/`
- Validation: VALIDATED
- Y5 Section 7.6.1 Condition 1 (distribution match): satisfied
- Y5 Section 7.6.1 Condition 2 (failure observability): satisfied
  (AUROC = 0.989)
- Y5 Section 7.6.1 Condition 3 (sufficient SNR): satisfied
  (n=15, t=6.76, p<0.001)

### Y3: Multi-agent MARL (6 comparisons)

- 6-pathway systematic investigation (v3 / v4 / v5 / v6 / v7 / v8)
- 5 of 6 pathways REFUTED at p<0.05
- 1 partial: v8 dlr_only (+0.06 at n=100, p_Bonf=0.0433)
- Source: `projects/project_f_multi_agent/`
- Y5 Section 7.6.1 Condition 1: violated (joint critic training drift)

### Y4: LLM self-monitoring (4 comparisons)

- 4 sample sizes: n=5 (stratified), n=20 (arith), n=100 (arith), n=20 (GSM8K 200-tok)
- All 4 REFUTED at p > 0.05
- Pre-reg kill switch: `STOP-PAPER-REFUTED-REVERSE`
- Source: `projects/project_g_llm_self_monitoring/`
- Y5 Section 7.6.1 Condition 2: weakly violated (AUROC ~ 0.50-0.65)

## S3. Y5 v1.3 §7.6 formal framework

See Y5 v1.3 master synthesis paper Section 7.6 for full details.
Summary:

- 7 Definitions (Auxiliary signal, Frozen reference, Policy distribution,
  3 Convergence Conditions, Transferability)
- 4 Propositions (P1 main theorem, P2 H10 consistency, P3 hybrid untested,
  P4 cross-task consistency)
- 4 Refutations (R1 non-stationary rescue, R2 LLM without retraining,
  R3 replication overturn, R4 LLM 7B/70B scale)
- Logical disjunction: F_falsified iff R1 OR R2 OR R3 OR R4
- Monotonicity: observing more Refutations forces STRONGER updates

NONE of R1-R4 has been observed across 11 empirical comparisons.

## S4. Cross-task meta-analysis (6 methods)

Section 5.3.1 + 5.3.2 of Y5 v1.3 applies 6 meta-analytic methods to
the 4 H10 sample-size p-values. All 6 agree: H10 is REFUTED.

| Method | Statistic | p | Conclusion |
|---|---|---|---|
| Fisher combined-p | chi^2 = 4.646, df = 8 | 0.7947 | NOT significant |
| Stouffer Z (equal weight) | Z = 1.105 | 0.135 (one-sided) | NOT significant |
| Stouffer Z (weighted sqrt(n)) | Z = 0.853 | 0.197 (one-sided) | NOT significant |
| Bonferroni min p | min p * 4 = 1.12 | alpha_bonf = 0.0125 | NOT rejected |
| Bonferroni-Holm step-down | 0/4 reject | alpha = 0.05 | NOT rejected |
| Hedges g (bias-corrected) | 4/4 CIs span zero | -- | NOT significant |

Forest plot: `papers/figures_v2/fig_h10_combined_p_forest.png`
SVG build source: `papers/figures_v2/_forest_plot_source.html`
JSON output: `experiments_log/_h10_combined_p.json`

## S5. Pre-registration chain

4 pre-registration documents, all dated BEFORE data aggregation:

1. `experiments_log/2026-07-28-PRE-REGISTERED-H10.md` (n=5 H10 pilot, kill switch at +0.05)
2. `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md`
   (n=20 GSM8K 200-tok follow-up, kill switch at +0.10)
3. `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md`
   (kill switch +0.05 -> +0.10 with power analysis justification)
4. `experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md`
   (Monitor + DLR hybrid test, n=100 seeds x 3 arms, GPU reservation
   2026-08-01 to 2026-08-15)

The pre-registration discipline is verified by checking that each
pre-reg file's date precedes the corresponding data aggregation date.

## S6. H10 empirical data (4 sample sizes)

For each of the 4 H10 sample sizes, the per-seed AUROC values are in
`experiments_log/_h10_n*_perseed.json`. The bootstrap CI / p-values
are in `experiments_log/_h10_n*_bootstrap.json`.

| Sample | n | F-J mean | 95% CI | Cohen's d | p |
|---|---|---|---|---|---|
| n=5 simple-arith (stratified) | 5 | -0.10 | [-0.45, +0.25] | -0.250 (post-hoc) | 0.6228 (Welch t) |
| n=20 simple-arith | 19 | +0.132 | [-0.079, +0.342] | +0.265 | 0.280 (bootstrap) |
| n=100 simple-arith | 98 | +0.015 | [-0.087, +0.117] | +0.030 | 0.787 (bootstrap) |
| n=20 GSM8K 200-tok | 19 | -0.053 | [-0.237, +0.158] | -0.120 | 0.714 (bootstrap) |

The n=5 row is marked post-hoc (R1.6, see Y5 Section 5.3.2 provenance note).

## S7. Y3 v8 dlr_only +0.06 result

The v8 dlr_only architecture (DLR cross-agent predicates in critic, no
Monitor) produces a small positive effect at n=100 paired seeds:

- Mean delta: +0.0617
- 95% CI: [+0.0084, +0.1149]
- Bonferroni-corrected p (3 arms): 0.0433
- Effect shrinks from +0.1447 at n=30 to +0.0617 at n=100 (consistent with
  regression-to-mean under diminishing novelty)

This is the only positive result in the 11-comparison record, and it
is NOT from the Monitor -- it is from the DLR predicates.

## S8. Code: pz_maddpg_v8.py (P3 training script)

The P3 hybrid pre-reg uses `pz_maddpg_v8.py` with 4 arms (one_verifier,
monitor_only, dlr_only, v8). The P3 pre-reg uses 3 arms (monitor_only,
dlr_only, v8).

Key code paths:
- `train_maddpg_v8(seed, n_updates, use_dlr_trust, use_dlr_critic)`:
  main training loop
- `extract_dlr_preds(obs)`: 24-dim DLR predicate vector from observation
- `run_random_baseline(seed, n_episodes)`: random policy baseline
- `Actor`, `Critic`, `TrustHead` networks (PyTorch)

The script was patched in v1.0 of the Y5 paper (this supplementary)
to support 4 arms. The patch added `monitor_only` arm via 3-line edit.

## S9. Code: r1_test.py (R1 test scaffold)

Tests R1 (Y5 Section 7.6.3): "A learned auxiliary signal that fails
Condition 1 (distribution match) but produces useful training signal in
non-stationary contexts."

Three arms:
- no_rescue: Monitor present, no policy reset
- periodic_reset: Monitor present, policy reset every 4 PPO updates
- no_monitor_reset: no Monitor, policy reset every 4 PPO updates (control)

The full R1 test (n=20 seeds x 3 arms x 200 PPO updates) is scheduled
for execution after the 1-hour wait period and the P3 hybrid
pre-reg completes.

## S10. Build chain (PDF generation)

Main paper Y5 v1.3 PDF (89 pages, 1.59 MB):
- Source: `papers/y5_monitor_transfer_synthesis.md` (140 KB)
- HTML: `papers/y5_v1_3_master_synthesis.html` (170 KB, pandoc 3.10.1)
- PDF: `papers/y5_v1_3_master_synthesis.pdf` (1.59 MB, Edge headless)

Companion paper PDFs:
- Y1 v1.0: 28 pages, 487 KB
- Y3 v1.0: 17 pages, 410 KB
- Y4 v0.6.1: PDF (pre-existing)

Build tool: `E:\gen_pdf.py` (Edge headless wrapper, no LaTeX needed)

## S11. OpenReview submission package

The COLM 2026 submission package is built by
`papers/_build_openreview_package.bat`:

- `arxiv_submission.tar.gz` (1.02 MB) -- main paper + cover letter + reviewer sim + S16
- `arxiv_submission_supplementary.tar.gz` (516 KB) -- supplementary + figures + JSONs + pre-regs
- `arxiv_checklist.txt` (1.3 KB) -- 14-item camera-ready checklist with SHA-256

The user uploads these to arXiv (with ARXIV_TOKEN) and to OpenReview
(separate submission).

## S12. H10 empirical chain (detailed)

For the H10 hypothesis (Decoupled Monitor transfers to LLM self-monitoring):

1. n=5 pilot (stratified split, 2026-07-29):
   - F-J = -0.10, Welch t = -0.516, p = 0.6228
   - Post-hoc analysis (not pre-registered)
   - Direction-consistent REFUTATION

2. n=20 simple-arith (2026-07-29):
   - F-J = +0.132, Cohen's d = +0.265
   - 95% CI [-0.079, +0.342]
   - 10000-resample bootstrap p = 0.280
   - NOT significant

3. n=100 simple-arith (2026-07-30):
   - F-J = +0.015, Cohen's d = +0.030
   - 95% CI [-0.087, +0.117]
   - 2000-resample bootstrap p = 0.787
   - NOT significant (near chance)
   - Required n for 80% power at d=+0.030: ~17,000 seeds (infeasible)

4. n=20 GSM8K 200-tok CoT (2026-07-31, v0.6.1):
   - F-J = -0.053, Cohen's d = -0.120
   - 95% CI [-0.237, +0.158]
   - 2000-resample bootstrap p = 0.714
   - NOT significant
   - Pre-reg kill switch: `STOP-PAPER-REFUTED-REVERSE`
   - Consistent direction across 2 task families (simple-arith + GSM8K 200-tok)

## S13. P3 hybrid pre-reg (in progress 2026-08-01)

The P3 hybrid pre-reg tests "Monitor + DLR > either alone" in
cooperative MARL:

- 3 arms: monitor_only, dlr_only, v8 (Hybrid)
- 60 jobs (20 seeds x 3 arms) in tightened config
- 300 jobs (100 seeds x 3 arms) in full pre-reg
- Execution: 2026-08-01 to 2026-08-15 (GPU reservation)
- Pre-reg: `experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md`

Bootstrap aggregator: `experiments_log/_agg_p3_hybrid.py`
Output JSON: `experiments_log/_p3_hybrid_bootstrap.json`

Preliminary (as of 2026-08-01 00:35):
- monitor_only: n=20 done, mean delta = +9.2, std = 4.2
- dlr_only: in progress
- v8: pending

## S14. References (7 new in v1.3)

Added to Y5 v1.3 bibliography:

1. Shimodaira 2000 -- "Improving predictive inference under covariate
   shift by weighting the log-likelihood function" (J. Stat. Plan. Inf.)
2. Cover & Thomas 1991 -- "Elements of Information Theory" (Wiley)
3. Valiant 1984 -- "A theory of the learnable" (CACM)
4. Haussler 1990 -- "Probably approximately correct learning" (AAAI)
5. Hanley-McNeil 1982 -- "The meaning and use of the area under a ROC
   curve" (Radiology)
6. Holm 1979 -- "A simple sequentially rejective multiple test
   procedure" (Scand. J. Stat.)
7. Hedges 1981 -- "Distribution theory for Glass's estimator of effect
   size and related estimators" (JEBS)

## S15. Compute and reproducibility

- Total compute: ~120 GPU-equivalent-hours on consumer hardware
  (RTX 4090 + i9-13900K, 64 GB RAM)
- P3 hybrid full pre-reg: ~50 GPU-h (2026-08-01 to 2026-08-15 window)
- R1 test: ~5-10 GPU-h (deferred)
- All experiments reproducible from .tex / .py / .json / .md in
  the git repository

## S16. Version history (v0.8 -> v1.3)

See `papers/supplementary_S16_version_history.md` for the complete
changelog. Summary:

- v0.8 (predecessor): 56 pages, no formal framework
- v1.0: 64 pages, added §7.6 framework, 12 reviewer items open
- v1.1: 70 pages, addressed 2 P0 items
- v1.2: 82 pages, addressed 10 items, added 6 meta-analytic methods
- v1.3: 89 pages, addressed 6 items, camera-ready Accept
- v1.3.1 (TBD): P3 hybrid result incorporated after 2026-08-15

## S17. Contact and contribution

- Liu Zewen (Archimedes Project, AGI-2026-001)
- Repository: https://github.com/aidless/agi-research
- HF README: `HF_README_P3.md` (for P3 launcher)

For questions or contributions, open an issue on the GitHub
repository.