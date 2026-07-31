# papers/figures_v2/ -- Y3/Y4 paper figures (committed)

Figures for the Y3 (6-pathway MA investigation) and Y4 (H10 LLM
self-monitoring) papers. These figures are referenced inline from
`papers/monitor_signal_vs_dlr_6pathway.md` and
`papers/project_g_v0_5_h10_paper.md`.

## Figures

### Y3 paper (`monitor_signal_vs_dlr_6pathway.md`)

| File | Section | What it shows | Data source |
|---|---|---|---|
| `y3_6pathway_summary.png` | end of 3.6 | Bar chart of all 6 pathways with mean paired difference vs no_verifier baseline; the only publishable result (v8 dlr_only) in red | `experiments_log/2026-07-29-y2-final-6-pathway.md` + `experiments_log/2026-07-29-v8-dlr-only-n100-aggregation.md` |
| `v5_vs_v8_shrinkage.png` | end of 3.6 | v5 effect-shrinkage from n=5 to n=212 (REFUTED) vs v8 dlr_only trajectory from n=5 to n=100 (SIG at all n); log-scale x axis | Y3 paper Section 3.3 (v5 table) and Section 3.6 (v8 dlr_only trajectory) |

### Y4 paper (`project_g_v0_5_h10_paper.md`)

| File | Section | What it shows | Data source |
|---|---|---|---|
| `h10_n5_forest.png` | 4.3 (after table) | n=5 per-arm mean differences for F-J, F-R, J-R with Welch t and p | n=5 hardcoded in `experiments_log/_mk_figures.py` |
| `h10_three_sample_arms.png` | after Abstract | 3-arm AUROC means (with SD bars) at n=5, n=20, n=100; chance line at 0.5 | n=5: hardcoded; n=20: `experiments_log/_h10_n20_summary.json`; n=100: `experiments_log/_h10_n100_bootstrap.json` |
| `y4_three_sample_summary.png` | after Abstract | Same 3 sample sizes, 3 paired contrasts, with cohens-d scaled xerr | Same JSON files as above |
| `h10_shrinkage_timeline.png` | end of 7.5 | F-J effect estimate at n=5, n=20, n=100 with 95% bootstrap CI; always crosses zero | `experiments_log/_h10_n20_bootstrap.json` + `experiments_log/_h10_n100_bootstrap.json` |

### Other (not embedded in markdown but useful as reference)

| File | Origin | Notes |
|---|---|---|
| `_h10_n20_forest.png` | `experiments_log/` | n=20 paired contrast forest; embedded in Y4 paper Section 7.2 via relative path |
| `_h10_n100_forest.png` | `experiments_log/` | n=100 paired contrast forest; embedded in Y4 paper Section 7.5 via relative path |

## Reproducing the figures

```bash
# Regenerate all figures_v2 PNGs from raw JSON
python experiments_log/_mk_figures.py
python experiments_log/_mk_more_figures.py
```

Both scripts depend on:
- `matplotlib` (already in `C:\Users\...\TRAE SOLO CN\...\python.exe`)
- `experiments_log/_h10_n20_summary.json`
- `experiments_log/_h10_n20_bootstrap.json`
- `experiments_log/_h10_n100_bootstrap.json`

## arXiv submission

The Y3 paper figures (5 PNGs) for arXiv live in
`papers/arxiv_submission/figures/` and are committed separately
for submission. The figures_v2 figures are NOT yet in the arXiv
tarball; if you want to submit the Y4 paper to arXiv too, run:

```bash
cd papers/arxiv_submission
cp ../figures_v2/h10_n5_forest.png figures/
cp ../figures_v2/h10_three_sample_arms.png figures/
cp ../figures_v2/y4_three_sample_summary.png figures/
cp ../figures_v2/h10_shrinkage_timeline.png figures/
# regenerate tar.gz
```
