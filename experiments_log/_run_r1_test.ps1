# R1 test launcher: Monitor in non-stationary cooperative MARL.
#
# Tests R1 (Y5 section 7.6.3): "A learned auxiliary signal that fails
# Condition 1 (distribution match) but produces useful training signal in
# non-stationary contexts."
#
# Three arms:
#   - no_rescue:        Monitor present, no policy reset (baseline)
#   - periodic_reset:   Monitor present, policy reset every 4 PPO updates
#   - no_monitor_reset: no Monitor, policy reset every 4 PPO updates (control)
#
# Smoke test: 1 seed x 3 arms x 8 PPO updates, ~30 sec
# Full test:  20 seeds x 3 arms x 200 PPO updates, ~30-50 min on CPU
#
# Requires pz_maddpg_v8.py to be patched to support re-initialization of
# actors and trust_heads mid-training (via init_actors, init_trust_heads
# parameters). This patch is documented in r1_test.py but not yet applied
# to pz_maddpg_v8.py. Run after the patch is applied.
#
# Usage: powershell -File _run_r1_test.ps1

$ErrorActionPreference = "Continue"
$PY     = "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\r1_test.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_r1_test_" + $TS + ".done")

"" | Out-File -FilePath $DONE -Encoding utf8
"=== R1 test launch $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Arms: no_rescue + periodic_reset + no_monitor_reset ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Per Y5 section 7.6.3: R1 hypothesis is non-stationary rescue by Monitor ===" | Out-File -FilePath $DONE -Append -Encoding utf8

# Use production config (20 seeds x 200 PPO updates)
$jobs = @()
foreach ($arm in @("no_rescue", "periodic_reset", "no_monitor_reset")) {
    foreach ($s in 0..19) {
        $jobs += @{arm = $arm; seed = $s}
    }
}

$MAX_PARALLEL = 6
$running = @()
$launched = 0

while ($launched -lt $jobs.Count) {
    while (($running.Count -lt $MAX_PARALLEL) -and ($launched -lt $jobs.Count)) {
        $job = $jobs[$launched]
        $arm = $job.arm
        $s = $job.seed
        $seedLog = Join-Path $LOGDIR ("_r1_test_" + $arm + "_s" + $s + ".log")
        $argList = @("-u", $SCRIPT, "--arm", $arm, "--seed", "$s",
                      "--n-updates", "200", "--reset-interval", "4")
        $proc = Start-Process -FilePath $PY -ArgumentList $argList `
            -RedirectStandardOutput $seedLog `
            -RedirectStandardError ($seedLog + ".err") `
            -PassThru -WindowStyle Hidden
        $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "LAUNCH  arm=$arm seed=$s pid=$($proc.Id) start=$start" | Out-File -FilePath $DONE -Append -Encoding utf8
        $running += @{proc = $proc; arm = $arm; seed = $s}
        $launched += 1
    }
    Start-Sleep -Seconds 30
    $stillRunning = @()
    foreach ($r in $running) {
        if ($r.proc.HasExited) {
            $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            "DONE    arm=$($r.arm) seed=$($r.seed) pid=$($r.proc.Id) exit=$($r.proc.ExitCode) end=$end" | Out-File -FilePath $DONE -Append -Encoding utf8
        } else {
            $stillRunning += $r
        }
    }
    $running = $stillRunning
    Write-Host ("Polling: launched={0}/{1} running={2} done={3}" -f $launched, $jobs.Count, $running.Count, ($launched - $running.Count))
}

while ($running.Count -gt 0) {
    Start-Sleep -Seconds 30
    $stillRunning = @()
    foreach ($r in $running) {
        if ($r.proc.HasExited) {
            $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            "DONE    arm=$($r.arm) seed=$($r.seed) pid=$($r.proc.Id) exit=$($r.proc.ExitCode) end=$end" | Out-File -FilePath $DONE -Append -Encoding utf8
        } else {
            $stillRunning += $r
        }
    }
    $running = $stillRunning
    Write-Host ("Final: running={0}" -f $running.Count)
}

"All $(($jobs.Count)) jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Aggregating R1 results ===" | Out-File -FilePath $DONE -Append -Encoding utf8