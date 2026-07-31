# What to Expect at Finalization

When `_finalize.py` runs (after the 60-job launcher finishes), it produces
the final state of the H10 v0.6.1 paper. This document explains what
each artifact contains and how to interpret it.

## TL;DR

1. Run `python papers/_finalize.py` from any cwd.
2. The script gates on the launcher's `.done` file having 60 DONE entries.
   If not yet done, it stops with a friendly message.
3. Once it runs, all 6 artifacts are updated atomically.

## Artifacts produced

### 1. `experiments_log/_h10_n20_gsm8k_bootstrap.json`

Format:
- `n_seeds_total` (60), `n_seeds_valid` (50-60 typically)
- `kill_switch_decision` (one of three pre-registered values)
- `kill_switch_rationale` (string explaining the decision)
- `mean_per_arm`: {"frozen": float, "joint": float, "random": float}
- `sd_per_arm`: same shape
- `contrasts`: {"F-J", "F-R", "J-R"} each with mean_diff, ci95, cohens_d,
   p_bootstrap_two_sided, sig_bonf (Bonferroni), required_n_for_80pct_power

### 2. `papers/figures_v2/forest_h10_n20_gsm8k.png`

3-arm forest plot for the GSM8K 200-token n=20 run. Mirrors the
existing n=20 simple-arith forest plot in style.

### 3. `papers/figures_v2/h10_shrinkage_timeline_v06.png`

4-point cross-task shrinkage timeline:
- n=5 (simple arith) (d=+0.275)
- n=20 (simple arith) (d=+0.265)
- n=100 (simple arith) (d=+0.030, chance level)
- n=20 (GSM8K 200-token) (d=newly measured)

Reads all 4 baseline JSONs dynamically; no hardcoded numbers.

### 4. `papers/project_g_v0_5_h10_paper.md`

Updated to v0.6.1 with §7.7-§7.9 filled in. Specifically:
- §7.7.4: per-seed results (placeholder pointing to logs).
- §7.7.5: aggregate table + 3 contrasts (F-J, F-R, J-R).
- §7.7.6: kill-switch verdict (one of 3 templates based on bootstrap).
- §7.7.8: cross-task combined verdict.
- §7.8: 4-row power re-analysis table.
- §7.9: updated final cross-task conclusion.
- Abstract: row 4 verdict updated to match the kill switch.

The pre-registered 3-option template (If F-J >= +0.10 / If F-J in [0,+0.10) /
If F-J < 0) is preserved as documentation.

### 5. `papers/y5_monitor_transfer_synthesis.md`

Updated to v0.6.1:
- §2.3 row 4 (GSM8K 200-tok n=20) filled with actual numbers.
- §3.1 row 5 (LLM self-monitoring GSM8K CoT) filled.
- §4.4 verdict block added with one of 3 conclusions based on kill switch.

### 6. `papers/HYPOTHESIS_STATUS.md`

H10 row updated to reflect the final verdict (e.g., "REFUTED on simple
arith AND GSM8K 200-tok"). The "in-flight" note becomes a "DONE" note.

## Interpreting the kill switch

| Decision | Meaning | Paper writeup |
|----------|---------|---------------|
| EXTEND-N50 | F-J >= +0.10 in [0, +0.10] | "warrants extending to n=50; see forthcoming Pre-Reg Amendment 2 / Section 7.10" |
| STOP-PAPER-REFUTED-AMBIGUOUS | F-J in [0, +0.10) | "REFUTED across two qualitatively different LLM tasks; stop at n=20; write paper" |
| STOP-PAPER-REFUTED-REVERSE | F-J < 0 | "REFUTED with consistent Joint > Frozen direction across both simple arithmetic and GSM8K" |

## Re-running

`_finalize.py` is idempotent: re-running on a finished run produces the
same output, EXCEPT that subsequent runs write the same numbers
(verifying idempotency). After the helpers run, the markdown has the
sentinel `<!--- END_VERDICT -->` between the verdict block and the
3-option template, which guarantees correct replacement on re-fill.

## If something goes wrong

If `_finalize.py` produces unexpected output:
1. Check `experiments_log/_h10_n20_gsm8k_*.log` files for per-job errors.
2. Re-run individual helpers to bisect:
   - `python experiments_log/_agg_h10_n20_gsm8k.py`
   - `python papers/_make_figures_v06.py`
   - `python papers/_fill_section_7_7.py`
   - `python papers/_fill_y5.py`
3. Check `papers/CHANGELOG.md` for recent changes.

## Before submitting to a venue

The .md is the canonical source; build the .pdf via pandoc:

```
pandoc papers\project_g_v0_5_h10_paper.md -o papers\project_g_v0_6_1_h10_paper.pdf
```

Verify all figures are present in `papers/figures_v2/`. Verify all
references resolve (file paths in §References point to existing files).
