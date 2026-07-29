# _run_v6_n30_3arm_r3.ps1 - v6 n=30 RERUN (40 jobs: 30 with_trusthead_random + 10 no_verifier s20-s29)
# 4-parallel, explicit python path with pettingzoo 1.24.3 (mpe submodule restored)
$ErrorActionPreference = "Continue"
$PY     = "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\pz_maddpg_v6.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_v6_3arm_n30_r3_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== v6 n=30 RERUN (with_trusthead_random 30 + no_verifier s20-s29, 4-parallel) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== After fixing pettingzoo 1.26.1 -> 1.24.3 (mpe submodule restored) ===" | Out-File -FilePath $DONE -Append -Encoding utf8

$jobs = @()
foreach ($s in 0..29) {
    $jobs += @{arm = "with_trusthead_random"; seed = $s}
}
foreach ($s in 20..29) {
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
        $seedLog = Join-Path $LOGDIR ("_v6_n30_r3_" + $arm + "_s" + $s + ".log")
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

"All 40 jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8
