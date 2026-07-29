# _run_v6_n5_3arm.ps1 - 3-arm n=5 ablation for pz_maddpg_v6
# 4-parallel, 15 jobs total (3 arms x 5 seeds)
$ErrorActionPreference = "Continue"
$PY     = "python"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\pz_maddpg_v6.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_v6_3arm_5seed_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== v6 3-arm 5-seed parallel (4-parallel) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8

# Jobs: (arm, seed) pairs
$jobs = @()
foreach ($arm in @("with_verifier", "no_verifier", "with_trusthead_random")) {
    foreach ($s in 0..4) {
        $jobs += @{arm = $arm; seed = $s}
    }
}

# Launch with 4-parallel
$MAX_PARALLEL = 4
$running = @()
$launched = 0

while ($launched -lt $jobs.Count) {
    # Fill up to MAX_PARALLEL
    while (($running.Count -lt $MAX_PARALLEL) -and ($launched -lt $jobs.Count)) {
        $job = $jobs[$launched]
        $arm = $job.arm
        $s = $job.seed
        $seedLog = Join-Path $LOGDIR ("_v6_" + $arm + "_s" + $s + ".log")
        $argList = @("-u", $SCRIPT, "--arm", $arm, "--seed", "$s",
                      "--n-updates", "80", "--n-episodes-per-update", "10",
                      "--n-eval-episodes", "15")
        $proc = Start-Process -FilePath $PY -ArgumentList $argList `
            -RedirectStandardOutput $seedLog `
            -RedirectStandardError ($seedLog + ".err") `
            -PassThru -WindowStyle Hidden
        $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "LAUNCH  arm=$arm seed=$s pid=$($proc.Id) start=$start" | Out-File -FilePath $DONE -Append -Encoding utf8
        $running += @{proc = $proc; arm = $arm; seed = $s}
        $launched += 1
    }

    # Check for completed jobs (poll every 30s)
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
    Write-Host ("Polling: launched={0} running={1} done={2}" -f $launched, $running.Count, ($launched - $running.Count))
}

# Wait for remaining
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

"All 15 jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8
