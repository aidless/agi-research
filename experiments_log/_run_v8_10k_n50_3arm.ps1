$ErrorActionPreference = "Continue"
# _run_v8_10k_n50_3arm.ps1 -- Proposition 3 hybrid pre-reg launcher (v1.3)
#
# Runs the 3-arm MADDPG test (Monitor alone / DLR alone / Hybrid) on
# Y3 cooperative multi-agent (simple_spread_v3). Per Pre-Reg:
#   experiments_log/2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md
#
# Compute reservation: ~50 GPU-h wall-clock on CPU-equivalent
# Execution window: 2026-08-01 to 2026-08-15
#
# Script layout: reuses pz_maddpg_v8.py with --arm {monitor_only, dlr_only, v8}
#   - monitor_only: use_dlr_trust=True, use_dlr_critic=False (NEW arm, see below)
#   - dlr_only    : use_dlr_trust=False, use_dlr_critic=True (existing arm)
#   - v8          : use_dlr_trust=True, use_dlr_critic=True (existing hybrid)
#
# NOTE: pz_maddpg_v8.py currently does not support the "monitor_only" arm.
#       To extend the script, change line 360 (the choices list) to:
#         choices=["v8", "no_verifier", "dlr_only", "monitor_only"]
#       and add a new use_dlr_trust condition:
#         use_dlr_trust = args.arm in ("v8", "monitor_only")
#       and update use_dlr_critic to:
#         use_dlr_critic = args.arm in ("v8", "dlr_only")
#       This is a 3-line edit; commit it BEFORE running this launcher.
#
# Usage: powershell -File _run_v8_10k_n50_3arm.ps1

$PY     = "C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe"
$SCRIPT = "E:\agi-research\projects\project_f_multi_agent\code\pz_maddpg_v8.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_v8_10k_n50_3arm_" + $TS + ".done")

"" | Out-File -FilePath $DONE -Encoding utf8
"=== Proposition 3 hybrid pre-reg 3-arm launcher (v1.3) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Arms: monitor_only + dlr_only + v8 (300 jobs total: 3 arms x 100 seeds) ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"=== Per Pre-Reg PROP3-HYBRID.md: ~50 GPU-h, execution window 2026-08-01 to 2026-08-15 ===" | Out-File -FilePath $DONE -Append -Encoding utf8

$jobs = @()
foreach ($arm in @("monitor_only", "dlr_only", "v8")) {
    foreach ($s in 0..99) {
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
        $seedLog = Join-Path $LOGDIR ("_v8_10k_n50_3arm_" + $arm + "_s" + $s + ".log")
        $argList = @("-u", $SCRIPT, "--arm", $arm, "--seed", "$s",
                      "--n-updates", "800", "--n-episodes-per-update", "10",
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

"All 300 jobs complete at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -FilePath $DONE -Append -Encoding utf8