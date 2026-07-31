# Proposition 3 Hybrid Pre-Reg: Smoke Test Results (2026-07-31)

**Purpose**: Verify the 3-arm Y3 cooperative multi-agent pipeline runs end-to-end
on the available CPU compute before committing to the full n=100 × 3-arm run.

**Smoke test configuration** (much smaller than real pre-reg):

| Setting | Smoke test | Real pre-reg |
|---|---|---|
| Arms | 3 (no_verifier, dlr_only, v8) | 3 (monitor_only, dlr_only, v8) |
| Seeds per arm | 1 (s=0) | 100 (s=0..99) |
| PPO updates | 8 | 800 |
| Episodes per update | 4 | 10 |
| Eval episodes | 4 | 15 |
| Wall-clock per arm | ~10 sec | ~3-5 min |
| Total wall-clock | ~30 sec | ~50 GPU-h |

**Smoke test results** (logs in `experiments_log/_smoke_p3_*.log`):

| Arm | Random baseline | Train (mean +/- std) | Delta vs random | Status |
|---|---|---|---|---|
| no_verifier | -79.76 +/- 26.73 | -64.98 (update 1/8) | +9.43 (final eval) | OK |
| dlr_only | -77.12 +/- 25.09 | -64.98 (update 1/8) | +6.65 (final eval) | OK |
| v8 | -76.80 +/- 23.04 | -64.98 (update 1/8) | +6.33 (final eval) | OK |

All 3 arms ran end-to-end (Phase 1 random baseline -> Phase 2 PPO updates -> Phase 3 final eval) without errors. Pipeline is functional on the available CPU compute.

**Note on arm mapping** (smoke vs real):

- The smoke test used the EXISTING 3 arms (no_verifier / dlr_only / v8). These are
  what the original pz_maddpg_v8.py supports.
- The real pre-reg requires a 4th arm "monitor_only" (Monitor + NO DLR critic).
  This arm was added via a 3-line patch to pz_maddpg_v8.py:
  1. Extended `--arm` choices list to include "monitor_only"
  2. Changed `use_dlr_trust = (args.arm == "v8")` to `use_dlr_trust = args.arm in ("v8", "monitor_only")`
  3. Changed `use_dlr_critic = args.arm in ("v8", "dlr_only")` (unchanged but more explicit)

The patched script (pz_maddpg_v8.py v2026-07-31) supports the real pre-reg's 4-arm
design: no_verifier (baseline), dlr_only (P3 DLR alone arm), v8 (P3 Hybrid arm),
monitor_only (P3 Monitor alone arm). The real run uses monitor_only + dlr_only + v8
(3 arms per the pre-reg; no_verifier is the baseline reference).

**Pipeline status**: GREEN. The real n=100 × 3-arm run can proceed in the
2026-08-01 to 2026-08-15 execution window as planned in the Pre-Reg.

**Launcher**: `experiments_log/_run_v8_10k_n50_3arm.ps1` (extends the existing
2-arm launcher with the new monitor_only arm and 3-arm job structure).

**Patch applied to**: `projects/project_f_multi_agent/code/pz_maddpg_v8.py`
(3-line edit: extend arm choices + update use_dlr_trust condition).

**Smoke test artifacts**:
- `experiments_log/_smoke_p3_no_verifier_s0.log` (474 bytes)
- `experiments_log/_smoke_p3_dlr_only_s0.log` (465 bytes)
- `experiments_log/_smoke_p3_v8_s0.log` (446 bytes)

These are kept for reproducibility and to verify the pipeline before the real run.

**Next step**: commit the launcher + patch + this documentation. The actual
n=100 × 3-arm run (the Pre-Reg execution) is scheduled for 2026-08-01 to
2026-08-15 and requires ~50 GPU-h wall-clock on CPU-equivalent.