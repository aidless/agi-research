$ErrorActionPreference = "Continue"
$PY     = "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe"
$SCRIPT = "E:\agi-research\projects\project_g_llm_self_monitoring\code\h10_real_pilot.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_h10_n20_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== H10 n=20 (3 arms x 20 seeds = 60 jobs) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Goal: confirm direction-consistent REFUTATION (Joint > Frozen) at larger n ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Each job: H10_N_TOTAL=8 rollouts, simple arithmetic, CPU, stratified ===" | Out-File -FilePath $DONE -Append -Encoding utf8

$jobs = @()
foreach ($arm in @("frozen", "joint", "random")) {
    foreach ($s in 100..119) {
        $jobs += @{arm = $arm; seed = $s}
    }
}

$MAX_PARALLEL = 1
$running = @()
$launched = 0

while ($launched -lt $jobs.Count) {
    while (($running.Count -lt $MAX_PARALLEL) -and ($launched -lt $jobs.Count)) {
        $job = $jobs[$launched]
        $arm = $job.arm
        $s = $job.seed
        $seedLog = Join-Path $LOGDIR ("_h10_n20_" + $arm + "_s" + $s + ".log")
        # Use env vars to control arm and seed
        $arm_env = switch ($arm) {
            "frozen" { "1" }
            "joint"  { "2" }
            "random" { "3" }
        }
        # Build a small wrapper script that sets the right env var
        $wrapper = Join-Path $LOGDIR ("_h10_n20_" + $arm + "_s" + $s + ".py")
        $wrapperContent = @"
import os, sys
sys.path.insert(0, r'E:\agi-research\projects\project_g_llm_self_monitoring\code')
os.environ['H10_SEED'] = '$s'
os.environ['H10_N_TOTAL'] = '8'
os.environ['H10_MAX_NEW_TOKENS'] = '16'
os.environ['H10_USE_SIMPLE'] = '1'
os.environ['H10_STRATIFIED'] = '1'
# Arm selection via env var (modify in the script or via flag)
os.environ['H10_ARM'] = '$arm'
sys.argv = ['h10_real_pilot.py', '--arm', '$arm']
exec(open(r'E:\agi-research\projects\project_g_llm_self_monitoring\code\h10_real_pilot.py').read())
"@
        Set-Content -Path $wrapper -Value $wrapperContent -Encoding UTF8
        $argList = @("-u", $wrapper)
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

"All 60 jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8
