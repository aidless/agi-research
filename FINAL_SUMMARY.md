# Archimedes Project -- Final Summary (2026-07-31)

A 5-year independent research program toward a self-improving
AGI substrate. Y0 Q3 (1st quarter of Year 0) is the current
milestone.

## Headline

The central architectural claim -- that decoupling the failure-
prediction Monitor from the policy gradient enables stable
self-monitoring in single-agent RL, and that this primitive
composes with slot-attention world models and fuzzy symbolic
verification -- is **PARTIALLY VERIFIED**.

- The single-agent RL claim is **VERIFIED** (H1, H2 with
  n=15 seeds, p<0.001, +39.5 mean improvement on LunarLander-v3).
- The "this primitive composes well" claim is **REFUTED** in
  multi-agent RL (H5, 5 of 6 pathways REFUTED) and LLM self-
  monitoring (H10, REFUTED at chance level at n=100).
- The "decoupling transfers" claim is **REFUTED** in any context
  outside single-agent RL.
- The one architectural lesson that DOES transfer across
  contexts is the **DLR (differentiable logic rules) cross-agent
  predicates in critic** pattern, which gives a small but
  statistically significant positive result in cooperative MARL
  (v8 dlr_only: +0.06 at n=100, p<0.05 with Bonferroni; 64/100
  seeds positive; 3-seed independent replication confirms
  direction).

## Honest interpretation

The Monitor decoupling architecture is a **context-specific
signal**, useful as a runtime guardrail in single-agent RL
(where it is verified) but not as a training signal in
multi-agent RL or LLM self-monitoring on simple arithmetic
tasks. The most defensible conclusion is that the simple
arithmetic trace is too coarse a signal for the Monitor
architecture to learn from, regardless of how the Monitor is
trained (frozen vs joint).

A future H10 study with a harder trace (e.g., GSM8K 200+ token
rollouts) is the only path to potentially validate the LLM
self-monitoring variant. The current evidence is sufficient
to conclude H10 is REFUTED for the simple arithmetic trace at
all sample sizes tested (n=5, 20, 100).

## What was done in this session

- **H10 n=100 follow-up** (300 jobs, 8h51m CPU) confirmed H10 is
  REFUTED at the chance level. All three arms (Frozen, Joint,
  Random) at n=100 are within +/- 0.02 of 0.5 (random). The
  n=20 direction reversal (Frozen > Joint by 0.13) was confirmed
  to be sampling noise, not a real effect.
- **v8 dlr_only n=100** (Y3) and **3-seed independent replication**
  (seeds 200, 201, 202) confirmed the only publishable positive
  result is direction-consistent and reproducible.
- **Y3 paper** updated with shrinkage trajectory and replication
  table. **Y4 paper** updated with the n=100 chance-level
  conclusion in the abstract and Section 7.5. **Y5 paper**
  updated with the n=100 and 3-seed data.
- **6 review fixes** (E1-E6) corrected critical data inconsistencies
  found across Y3 and Y4 papers (thesis Section 26.7 CI typo,
  v8 n=30 mean alignment, Section 4.3 contradiction, Section
  5.1/5.2 narrative update, Section 7.6 power re-analysis).
- **Supplementary materials** expanded from S1-S5 to S1-S15
  with full provenance of every reported number, all figure data
  sources, and a canonical reproduce script.
- **8 figures** (5 H10 + 2 Y3 + 1 thesis global) generated and
  embedded in the appropriate papers.
- **H1-H10 status table** created as the canonical reference
  for all hypothesis verdicts.
- **AAMAS 2027** (Y3) and **COLM 2026** (Y4) cover letters
  written, both including the latest n=100 and 3-seed replication
  data.
- **Thesis v2.0** compiled to 67-page PDF with all the new data
  and a global decoupling-across-contexts figure.
- **Y1, Y3, Y4, Y5 papers all have PDF versions** (markdown +
  generated LaTeX + pdflatex compile).

## Session summary

- **18 commits this session**, all pushed to origin/main
- **0 tokens spent** on arXiv upload (still requires ARXIV_TOKEN)
- **0 issues remaining** that block the arXiv upload of either
  paper (Y3 or Y4) once a token is provided
- **All 4 paper PDFs** (Y1, Y3, Y4, Y5) are committed and
  reproducible from the markdown sources

## Files in the public release

| Path | Size | Description |
|---|---|---|
| README.md | 4.5KB | Project overview + key results |
| CHANGELOG.md | 1.6KB | Major version milestones |
| FINAL_SUMMARY.md | this file | Headline + session summary |
| PROGRESS.md | 130KB | Session-by-session log |
| papers/HYPOTHESIS_STATUS.md | 3.3KB | H1-H10 status table |
| papers/monitor_signal_vs_dlr_6pathway.md + .tex + .pdf | 23KB + 30KB + 282KB | Y3 paper |
| papers/project_g_v0_5_h10_paper.md + .tex + .pdf | 20KB + 30KB + 426KB | Y4 paper |
| papers/y5_monitor_transfer_synthesis.md + .tex + .pdf | 13KB + 13KB + 145KB | Y5 paper |
| papers/y1_9hypothesis_framework.md + .tex + .pdf | 18KB + 16KB + 176KB | Y1 framework |
| papers/supplementary_materials.md | 24KB | S1-S15 reproducibility |
| papers/VENUE_PLAN.md | 2.5KB | Y3 -> AAMAS 2027, Y4 -> COLM 2026 |
| papers/cover_letter_aamas2027.md | updated | AAMAS 2027 cover |
| papers/cover_letter_colm2026.md | 5KB | COLM 2026 cover (new) |
| papers/REPRODUCE.sh | 2.4KB | Y3 reproduce script |
| papers/figures_v2/ | 280KB total | 7 PNGs + README + tar.gz |
| papers/arxiv_submission/ | 720KB | Y3 arXiv package (dry-run OK) |
| thesis_draft_v2.0.tex + .pdf | 156KB + 471KB | 67-page thesis |
| experiments_log/_h10_n{20,100}_bootstrap.json | <2KB each | H10 paired bootstrap |
| experiments_log/_v8_sanity_4seed.json | 1.5KB | v8 3-seed replicate |

## Citation

Liu, Z. (2026). The Archimedes Project: A 5-Year Independent
Research Program Toward a Self-Improving AGI Substrate.
Independent Research Report.
