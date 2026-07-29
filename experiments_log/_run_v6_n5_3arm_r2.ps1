$ErrorActionPreference = "Continue"
$PY     = "python"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\pz_maddpg_v6.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_v6_3arm_5seed_r2_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== v6 3-arm 5-seed RERUN (4-parallel) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== After fixing use_random_trust_input trust-head branch bug ===" | Out-File -FilePath $DONE -Append -Encoding utf8

$jobs = @()
foreach ($arm in @("with_verifier", "no_verifier", "with_trusthead_random")) {
    foreach ($s in 0..4) {
        $jobs += @{arm = $arm; seed = $s}
    }
}

$MAX_PARALLEL = 4
$running = @()
$launched = 0

while ($launched -lt $jobs.Count) {
    while (($running.Count -lt $MAX_PARALLEL) -and ($launched -lt $jobs.Count)) {
        $job = $jobs[$launched]
        $arm = $job.arm
        $s = $job.seed
        $seedLog = Join-Path $LOGDIR ("_v6_r2_" + $arm + "_s" + $s + ".log")
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
