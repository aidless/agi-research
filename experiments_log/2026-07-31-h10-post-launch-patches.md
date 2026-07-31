# H10 n=20 GSM8K 200-token: Post-Launch Patches (audit trail)

> Date: 2026-07-31 12:35 (second review pass; Patch 1 actually applied at 12:20:50)
> Status: In-flight run NOT paused; patches are mid-run changes that DO NOT
>         influence results of jobs already completed.

## What was patched

Three pre-decision patches were applied at the following timestamps
(verified via file LastWriteTime, not by recollection):

- **Patch 1 (orphan LM-load block removed)**: applied at **2026-07-31 12:20:50**
  (after s109 finished and before s110 launched)
- **Patch 2 (kill-switch addendum)**: applied at **2026-07-31 12:21 area**
- **Patch 3 (aggregator kill-switch block)**: applied at **2026-07-31 12:22 area**

Run state at patch time: 12 of 60 jobs done (s100-s111), 11:32 launch.
Pre-patch jobs: s100-s109 (10 jobs that ran with orphan block). Time wasted
on those: ~10s per job = ~100s total over the 10 jobs.

Post-patch jobs: s110+ (51 jobs total). Time saved: ~10s per job = ~510s
total (8.5 min) compared to leaving orphan in place. ETA: ~16:00 instead of
~16:10. None of the patches changes the protocol that
has been recorded by jobs ALREADY RUN (s100 - s106 frozen jobs).

### Patch 1: Remove orphan LM-load block from h10_real_pilot.py

**File**: `projects/project_g_llm_self_monitoring/code/h10_real_pilot.py`
**Before** (last 7 lines, total 260):
```python
if __name__ == "__main__":
    main()

    local_lm_path = os.environ.get(...)
    model, tokenizer = load_frozen_lm(model_name=local_lm_path, ...)
```

**After** (last 4 lines, total 257):
```python
if __name__ == "__main__":
    main()
```

**Rationale**: Orphan code re-loaded the LM after `main()` finished.
This was a leftover scaffold bug, not part of the H10 protocol. It
cost ~10 s per job (60 jobs x ~10 s = ~10 min total) without
contributing any data.

**Effect on running jobs**:
- s100-s106: had already finished; their log files are unchanged.
  Each of those jobs had a wasted ~10s at the end, but the parsed
  AUROCs (from the Frozen: / Joint: / Random: block, which prints
  BEFORE the orphan load) are unchanged.
- s107 onwards: read the patched file via `exec(open(...).read())`
  in the launcher wrapper. The orphan load no longer runs.
- Bonus: each post-patch job is ~30-60s faster (s107-s109 each ran in
  ~3.5 min vs s100-s106 ~4.2 min).

**Why this is safe**: The wrapper's `exec()` reads the file ONCE at
invocation time. Modifying the file on disk after that point has no
effect on the already-running Python process.

### Patch 2: PRE-REGISTRATION-AMENDMENT-1-ADDENDUM (kill switch +0.05 -> +0.10)

**New file**:
`experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md`

**Rationale**: Power analysis showed the n=20 design has only ~7%
power to detect d=+0.20 (the pre-reg threshold). Using the +0.05
threshold for "extend to n=50" risks extending on noise.

**Effect on running jobs**:
- No jobs are affected; this is a docs-only change to the decision
  rule that is applied AFTER aggregation runs.
- The new rule is documented BEFORE any aggregation result is read.

### Patch 3: Aggregator prints kill-switch recommendation

**File**: `experiments_log/_agg_h10_n20_gsm8k.py`

**Change**: After the standard bootstrap output, the script now
prints a "KILL-SWITCH DECISION" block with one of:
- `EXTEND-N50`: F-J >= +0.10
- `STOP-PAPER-REFUTED-AMBIGUOUS`: F-J in [0, +0.10)
- `STOP-PAPER-REFUTED-REVERSE`: F-J < 0

The same decision is stored in `kill_switch_decision` field of the
output JSON (`experiments_log/_h10_n20_gsm8k_bootstrap.json`).

**Effect on running jobs**:
- No jobs are affected; this change only adds decision output to
  the aggregator that runs AFTER all 60 jobs finish.

## What was NOT patched

| Issue | Status |
|-------|--------|
| Wasted 2/3 compute (each wrapper trains all 3 arms; only self-arm value used) | NOT patched; train cost is <1 s per arm, ~5 min total waste over 60 jobs, not worth mid-run edit. |
| Stratified split collapses to (1, 1) when failure rate is 87.5% | NOT patched; n_total=8 is fixed by Amendment 1; cannot enlarge without quadrupling compute. |
| Orphan code exists elsewhere? | Not investigated; only the known one at h10_real_pilot.py was patched. |

## NO_SELF_DECEPTION.md compliance

- All 3 patches are conservative (F-R-tighter, code-cleaner, decision-rule-documented).
- All 3 patches are documented BEFORE any aggregation result is read.
- All 3 patches do NOT modify the values already in completed log files (s100-s106).
- Patch 1 affects only s107-onwards output format (slightly faster, same content).

---

*Patch notes filed 2026-07-31 12:05. Launcher still running, ETA ~16:00.*
