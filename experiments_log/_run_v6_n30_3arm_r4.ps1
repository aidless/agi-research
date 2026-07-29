# _run_v6_n30_3arm_r4.ps1 - v6 n=30 CLEAN RERUN (50 jobs, all with same python/pettingzoo 1.24.3)
# 30 with_verifier (s0-s29) + 20 no_verifier (s0-s19) -- remove the n=30 batch env-inconsistency confound
$ErrorActionPreference = "Continue"
$PY     = "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\pz_maddpg_v6.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_v6_3arm_n30_r4_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== v6 n=30 CLEAN RERUN (50 jobs: 30 with_verifier + 20 no_verifier s0-s19) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Explicit python path: pettingzoo 1.24.3 (mpe restored) ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Goal: remove env-inconsistency confound from n=30 r2/r3 batch ===" | Out-File -FilePath $DONE -Append -Encoding utf8

$jobs = @()
foreach ($s in 0..29) {
    $jobs += @{arm = "with_verifier"; seed = $s}
}
foreach ($s in 0..19) {
    $jobs += @{arm = "no_verifier"; seed = $s}
}

$MAX_PARALLEL = 4
$running = @()
$launched = 0

while ($launched -lt $jobs.Count) {
    while (($running.Count -lt $MAX_PARALLEL) -and ($launched -lt $jobs.Count)) {
        $job = $jobs[$launched]
        $arm = $job.arm
        $s = $job.seed
        $seedLog = Join-Path $LOGDIR ("_v6_n30_r4_" + $arm + "_s" + $s + ".log")
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
    Start-Sleep -Seconds 60
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
    Start-Sleep -Seconds 60
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

"All 50 jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8
