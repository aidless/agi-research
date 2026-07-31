# P3 Hybrid Pre-Reg Live Monitoring Dashboard

**Updated:** 2026-08-01 00:42:40 (session day 1)
**Launcher:** experiments_log/_run_p3_hybrid_production.ps1
**Started:** 2026-08-01 00:05:57
**ETA for completion:** ~2026-08-01 01:25-01:30 (based on monitor_only ETA of 00:35)

## Progress summary

| Time | monitor_only | dlr_only | v8 (Hybrid) | Total |
|---|---|---|---|---|
| 00:05 (start) | 0/20 | 0/20 | 0/20 | 0/60 |
| 00:30 | 20/20 (DONE) | 4/20 | 0/20 | 24/60 (40%) |
| 00:35 | 20/20 | 5/20 | 0/20 | 25/60 (42%) |
| 00:40 | 20/20 | 7/20 | 0/20 | 27/60 (45%) |
| 00:42 (now) | 20/20 | 7/20 | 0/20 | 27/60 (45%) |
| 01:00 (target) | 20/20 | 12/20 | 0/20 | 32/60 (53%) |
| 01:30 (est done) | 20/20 | 20/20 | 20/20 | 60/60 (100%) |

## monitor_only (DONE)

- All 20 seeds completed
- Mean delta vs random: **+8.4 to +9.2** (positive)
- Per-seed log files: experiments_log/_p3_hybrid_monitor_only_s{0..19}.log
- This is a POSITIVE signal for the Monitor architecture in the Y3
  cooperative multi-agent setting. Contrast with the v0.x Y3 results
  (5/6 REFUTED) which used different arm configurations.

## dlr_only (IN PROGRESS)

- 7 / 20 seeds completed as of 00:42
- Mean delta vs random: (preliminary, not aggregated yet)
- Historical baseline from v8 dlr_only (n=100): +0.0617

## v8 (Hybrid) (PENDING)

- 0 / 20 seeds completed
- Expected to start ~01:00-01:05 (after dlr_only batch 1 completes)

## Verdict (preliminary)

The pre-reg kill switch decision rule is:
- VALIDATED if v8 - dlr_only >= +0.05 with p<0.05 (Bonferroni-corrected)
- REFUTED if v8 - dlr_only < +0.05 OR p >= 0.05
- EXTEND (run full n=100 pre-reg) if borderline

**Provisional verdict** (subject to change when all 60 jobs complete):

If monitor_only is +9 and dlr_only is ~+6 (historical), then v8 needs to be
>=+11 to satisfy the +0.05 threshold for VALIDATION. If v8 is ~+9 (similar
to monitor_only alone), P3 is REFUTED. The framework's prior is
REFUTED (P3 was a Proposition, not a Theorem), and the preliminary
monitor_only positive is consistent with either verdict.

## Aggregator

- Script: experiments_log/_agg_p3_hybrid.py
- Output: experiments_log/_p3_hybrid_bootstrap.json
- Re-run anytime: `python experiments_log/_agg_p3_hybrid.py`
- Last run: 2026-08-01 00:42:11 (n=24 done; verdict INSUFFICIENT_DATA
  since dlr_only and v8 not yet done)

## Logs to monitor

```
experiments_log/_p3_launcher_master.log          # main launcher progress
experiments_log/_p3_hybrid_<arm>_s<seed>.log     # per-job final eval
experiments_log/_p3_hybrid_bootstrap.json         # aggregator output
experiments_log/_p3_hybrid_<TS>.done             # completion marker (when all done)
```

## Next steps after P3 completes (~01:30)

1. Re-run `_agg_p3_hybrid.py` for final verdict
2. Update Y5 v1.3 -> Y5 v1.3.1 with P3 result
3. Re-render Y5 v1.3.1 PDF / DOCX / HTML via gen_pdf.py
4. Re-run reviewer simulator on v1.3.1 (expected: Accept with 0-2 items)
5. Update COLM 2026 cover letter (if P3 changes abstract or conclusions)
6. Commit Y5 v1.3.1 to git + push to origin
7. Begin R1 test execution (deferred per user instruction, target after 1-hour mark)