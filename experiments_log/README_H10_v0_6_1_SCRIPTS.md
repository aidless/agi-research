# H10 GSM8K 200-Token Follow-up: Scripts Index

This directory contains the scripts and intermediate files for the
H10 v0.6.1 GSM8K 200-token follow-up to Project G. The 60-job
launcher (`_run_h10_n20_gsm8k.ps1`) runs sequentially and produces
per-job logs that the aggregator below consumes.

## Pipeline order

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 1. Smoke validation | `2026-07-31-h10-gsm8k-200t-validation.md` | n/a | validation write-up |
| 2. Mid-run diagnostic | `_diag_h10_n20_gsm8k.py` | `_h10_n20_gsm8k_*.log` | stdout (per-run snapshot to log file) |
| 3. Post-launch aggregation | `_agg_h10_n20_gsm8k.py` | `_h10_n20_gsm8k_*.log` | `_h10_n20_gsm8k_bootstrap.json` |
| 4. Figures | `_make_figures_v06.py` (in `papers/`) | bootstrap JSONs (n=5/20/100 simple arith + GSM8K) | `papers/figures_v2/{forest_h10_n20_gsm8k, h10_shrinkage_timeline_v06}.png` |
| 5. Y4 paper fill | `_fill_section_7_7.py` (in `papers/`) | bootstrap JSON | `papers/project_g_v0_5_h10_paper.md` |
| 6. Y5 paper fill | `_fill_y5.py` (in `papers/`) | bootstrap JSON | `papers/y5_monitor_transfer_synthesis.md` |
| 7. HYPOTHESIS_STATUS update | (inside `_finalize.py`) | bootstrap JSON | `papers/HYPOTHESIS_STATUS.md` |
| 8. Full sequence (1-7) | `_finalize.py` (in `papers/`) | launcher .done file + bootstrap JSON | all of the above |

After step 8, the user rebuilds the .pdf via pandoc:

```
pandoc papers\project_g_v0_5_h10_paper.md -o papers\project_g_v0_6_1_h10_paper.pdf
```

## Per-run snapshots

`_diag_h10_n20_gsm8k_*.log` files contain the diagnostic output
captured at each checkpoint during the launcher run.

## JSON output format

`_agg_h10_n20_gsm8k.py` produces `_h10_n20_gsm8k_bootstrap.json`
matching this schema (one entry per contrast, two-sided bootstrap
95% CIs):

```json
{
  "n_seeds_total": int,
  "n_seeds_valid": int,
  "kill_switch_decision": "EXTEND-N50" | "STOP-PAPER-REFUTED-AMBIGUOUS" | "STOP-PAPER-REFUTED-REVERSE",
  "kill_switch_rationale": str,
  "mean_per_arm": {"frozen": float, "joint": float, "random": float},
  "sd_per_arm": {"frozen": float, "joint": float, "random": float},
  "contrasts": {
    "F-J": {"mean_diff": float, "ci95": [lo, hi], "cohens_d": float,
            "p_bootstrap_two_sided": float, "sig_bonf": bool,
            "required_n_for_80pct_power": float, "n": int},
    "F-R": {...},
    "J-R": {...},
  },
  "alpha_bonferroni": 0.0167,
  "method": "paired seed-level t-test + 2,000-replicate bootstrap 95% CI",
}
```

## Pre-reg / kill-switch chain

- Pre-Registration Amendment 1: `2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md`
- Kill-switch addendum (tightens +0.05 -> +0.10): `2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md`
- Mid-run patching notes: `2026-07-31-h10-post-launch-patches.md`
- This index: `README_H10_v0_6_1_SCRIPTS.md`

## Decision rule (Pre-Reg Amendment 1 addendum)

| Frozen - Joint (n=20) | Decision |
|---|---|
| `>= +0.10` | EXTEND to n=50 |
| `[0, +0.10)` | STOP-PAPER-REFUTED-AMBIGUOUS |
| `< 0` | STOP-PAPER-REFUTED-REVERSE |

The decision is computed automatically by `_agg_h10_n20_gsm8k.py`
based on the bootstrap JSON's `contrasts['F-J']['mean_diff']` value.

## NO_SELF_DECEPTION.md compliance

- Pre-reg filed BEFORE data collection: 2026-07-31 12:21 (Amendment 1).
- Kill switch recorded BEFORE aggregation: 2026-07-31 12:22 (Addendum).
- Negative control (Random Monitor) included in every run.
- No seed silently dropped; NaN seeds masked explicitly in `mask`.
- No post-hoc decision rule modification.
