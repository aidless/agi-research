# STATUS

**STATUS: WORK-IN-PROGRESS — not A yet.**

This ledger is the front door for the notebook. It sorts statements into three kinds. It does not add a new experiment.

- **Verified** means the figure was recomputed from a JSON file in this repository, and the arithmetic matches that file. It does not mean another lab, a venue, or a reviewer has checked it.
- **Note** means a lab-notebook entry. The per-seed files named in the note are missing, the note disagrees with itself, or a later entry revised it.
- **Claim** means draft prose, a cover letter, or a reviewer-simulator line that the committed JSON does not support, or a publication status with no external record in this tree.

## Gap list

1. Nothing here is peer reviewed. The tree has no acceptance letter and no arXiv identifier. `papers/reviewer_simulator_output_*.md` is a simulator.
2. Drafts still cite a single-agent headline, t = 6.76 and p < 0.001, as H1 and H2 validated. That figure is not recoverable from the committed 15-seed list. The 2026-07-28 pre-registered control says the monitor-versus-random contrast was not supported. This PR does not rewrite those drafts.
3. The frozen-versus-joint AUROC note (mean delta 0.724, LunarLander-v3, five seeds) has no checkpoint files in the tree. The multi-game falsifier named in that note (12 or more games) was not run.
4. Multi-agent monitor pathways stay **falsified** as labeled in `experiments_log/2026-07-29-y2-final-6-pathway.md`. That file's two opening paragraphs disagree on whether the DLR-only arm is a positive result. The n = 100 aggregation is markdown only. The only committed numeric replicate is three seeds and is not significant.
5. H10 stays **falsified**: the pre-registered success rule was not met. The same JSON intervals include zero, so the log also does not prove a significant effect in the other direction.
6. Push CI does not re-run training. The string audit exits 0 when a quoted number is absent. Four paper PDF jobs are manual-only. There is no training unit-test suite.
7. Author-machine absolute paths remain in 119 files (327 occurrences) of dated drafts, logs, launch scripts, and some training code, counted when this page was written. This pass removed them from the front door, `AGENTS.md`, `PROGRESS.md`, `prompts/`, and `projects/project_a_self_improvement/code/README.md`.
8. `PUBLICATION_HOLD.md` contains both an active-hold section and a later lifted section. Neither is an acceptance.
9. A predicate-accuracy table labeled validated for H3 was not recomputed here. It is a different object from `experiments_log/h3_500k_summary.json`, which shows the real monitor behind a random control at 500k steps.
10. Grade A would need a new run of the single-agent contrast under the negative-control rule, or a corpus that stops citing the unrecovered t-statistic as a result. This PR does neither.

## Verified from committed JSON

| Record | What the file supports | File |
|---|---|---|
| H10 simple arithmetic, **falsified** | 98 valid seeds, not 100. Frozen minus joint mean difference +0.0153, Cohen's d +0.030, 95% bootstrap interval [-0.087, +0.117], two-sided bootstrap p = 0.787. The interval includes zero. d matches mean difference / sd of differences. | `experiments_log/_h10_n100_bootstrap.json` |
| H10 GSM8K, 200-token, **falsified** | 19 valid seeds of 20. Frozen minus joint −0.0526, d −0.120, interval [−0.237, +0.158], p = 0.714. The lab note's kill-switch label is `STOP-PAPER-REFUTED-REVERSE`. The interval includes zero. | `experiments_log/_h10_n20_gsm8k_bootstrap.json` |
| H10 simple arithmetic, n = 20 file, **falsified** | 19 seeds. Frozen minus joint +0.132, d +0.265, interval [−0.079, +0.342], bootstrap p = 0.28. Drafts that quote p = 0.2623 do not match this file. | `experiments_log/_h10_n20_bootstrap.json` |
| Y1.3 monitor versus random shaping, not supported | Per-seed list: real n = 15 mean 80.11, random n = 10 mean 66.53, difference +13.58, Welch t = 0.781. The file sets `h1_verdict.supported` to false. This matches `experiments_log/2026-07-28-phase15-y13-n15-final.md`. | `experiments_log/y13_n10_per_arm_summary.json` |
| Y1.3 real versus PPO, not the headline t-test | Same seeds versus PPO n = 5 mean 40.64: difference +39.47, Welch t = 1.938. A one-sample t of the 15 seeds against that PPO mean is 3.33, with 12 of 15 seeds above the mean. | same file |
| H3 at 500k steps, real behind random | Real mean 170.74 (n = 5) versus random mean 223.87 (n = 5), difference −53.13, stored Welch t = −2.16. | `experiments_log/h3_500k_summary.json` |
| v8 sanity replicate, not a confirmation | The file is named `4seed` and its prose says four seeds. The numeric fields are three seeds (200, 201, 202), paired differences +0.27, −0.08, +0.30, mean 0.163, t = 1.34, normal-approximation p = 0.36. | `experiments_log/_v8_sanity_4seed.json` |

## Notes (not rechecked seed-by-seed)

### Multi-agent monitor pathways — falsified

`experiments_log/2026-07-29-y2-final-6-pathway.md` labels the monitor pathways refuted. That label stays. The per-seed checkpoint JSON for those arms is not in the tree, so the p-values in the note are not verified here. Do not relabel the monitor pathways as a positive result.

The v8 DLR-only n = 100 write-up (`experiments_log/2026-07-29-v8-dlr-only-n100-aggregation.md`) is a note: mean paired difference +0.0617 on a return near −69.65. No per-seed JSON for that aggregation is in the tree. It is a hand-crafted predicate in the critic, not the monitor hypothesis. The three-seed file above does not confirm it.

### Frozen versus joint AUROC

`experiments_log/2026-07-25-joint-ablation-A.md` reports frozen mean AUROC 0.796, joint mean 0.072, mean delta 0.724, five of five seeds in the same direction, on LunarLander-v3 only. Seed 2's frozen AUROC in that table is 0.21. The hypothesis note reports a Wilcoxon p of 0.0625. The checkpoint paths named in the log are not in the tree.

### Other single-agent rows

H2.0 at n = 10 per arm is a note of "not supported" (`experiments_log/2026-07-28-H2.0-n10-final.md`): forward-model delta +32.9, t = 1.60; MLP shaping delta +29.1, t = 1.38. H4 is a one-seed AUROC comparison in `papers/y1_9hypothesis_framework.md`. H1.4 exploration bonus is labeled refuted in that same hypothesis file.

## Claims that this page does not carry

- t = 6.76, df = 14, p < 0.001, and "13/15 seeds positive", used as H1 and H2 validated in `papers/HYPOTHESIS_STATUS.md` and the paper drafts. The committed seed list does not produce that t or that count. `experiments_log/2026-07-28-DEC-Y1.3-v1.1-publishability.md` calls the comparison the wrong baseline. `docs/NO_SELF_DECEPTION.md` records the same retraction.
- "Camera-ready", "submitted to AAMAS 2027", "submitted to COLM 2026", and reviewer-simulator "Accept". Those are draft words. `thesis_v3_0_integration_plan.md` also talks about a future acceptance. No external decision is in the tree.
- A CI badge as evidence that training results were reproduced. The badge was removed. The workflow section in `README.md` is the configured behavior.

`papers/HYPOTHESIS_STATUS.md` is a 2026-07-31 snapshot. Its H1 and H2 rows still say validated. Use this file, not that table, as the current ledger. H5 and H10 in that snapshot stay falsified; this file does not reverse them.
