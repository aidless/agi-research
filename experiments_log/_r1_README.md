# R1 test launcher: ready to execute after 1-hour mark.

This script launches the R1 test (3 arms x 20 seeds x 200 PPO updates)
after the 1-hour waiting period. The launcher waits until 01:05 (1 hour
from session start) before firing.

## Background

- 1-hour waiting period per user instruction (started 2026-08-01 00:05)
- P3 hybrid pre-reg running in parallel (will be ~30/60 by 01:00)
- R1 test reuses pz_maddpg_v8.py + r1_test.py

## Run command

```powershell
# Wait until 01:05, then run
Start-Sleep -Seconds (([DateTime]'01:05:00' - [DateTime]::Now).TotalSeconds)
& "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe" `
  "E:\agi-research\experiments_log\_run_r1_test.ps1"
```

Or use the wrapper script:

```powershell
& "E:\run_r1_after_wait.ps1"
```

## Expected timeline

- 01:00: dlr_only done, v8 starts (estimated)
- 01:05: 1-hour mark, R1 test launches (in parallel with v8)
- 01:30-01:40: P3 hybrid all 60 jobs done; FINAL VERDICT
- 01:30-01:50: R1 test all 60 jobs done (30-50 min CPU)

## Aggregator

After R1 completes, run:
```bash
python experiments_log/_agg_r1_test.py
```

This will output R1 verdict: did periodic_reset beat no_monitor_reset?
If yes, R1 (Monitor rescues in non-stationary contexts) is observed.

## Bootstrap JSON

R1 results will be in:
- experiments_log/_r1_test_<arm>_s<seed>.log (per-job logs)
- experiments_log/_r1_test_<TS>.done (completion marker)
- experiments_log/_r1_bootstrap.json (aggregator output)

## Git workflow

After R1 completes:
1. Update R1 results in Y5 v1.3.x
2. Decide: does the R1 result update the framework's R1 prediction?
3. If R1 observed: framework updates (Condition 1 is not strictly necessary)
4. If R1 NOT observed: framework survives (R1 remains open)