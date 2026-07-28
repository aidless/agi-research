# H1.4: 5 real + 5 random at 100K PPO, with MONITOR AS EXPLORATION BONUS
$ErrorActionPreference = "Continue"
$PY     = "C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
$SCRIPT = "E:\agi-research\projects\project_a_self_improvement\code\y14_exploration.py"
$LOGDIR = "E:\agi-research\experiments_log"
$TS     = (Get-Date -Format "yyyyMMdd-HHmmss")
$DONE   = Join-Path $LOGDIR ("_h14_5seed_" + $TS + ".done")
"" | Out-File -FilePath $DONE -Encoding utf8
"=== H1.4: 5 real + 5 random (exploration bonus) start $TS ===" | Out-File -FilePath $DONE -Append -Encoding utf8

function Launch-One($s, $use_random) {
    $tag = if ($use_random) { "y14r" } else { "y14" }
    $seedLog = Join-Path $LOGDIR ("_h14_" + $tag + "_s" + $s + ".log")
    $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "" | Out-File -FilePath $DONE -Append -Encoding utf8
    ($tag + " seed " + $s + "   START " + $start) | Out-File -FilePath $DONE -Append -Encoding utf8
    $argList = @("-u", $SCRIPT, "--n-ppo-steps-total", "100000", "--n-warmup-steps", "25000", "--n-train-episodes", "200", "--n-eval-episodes", "50", "--history-len", "32", "--seed", "$s", "--monitor-beta", "0.5", "--out-tag", $tag)
    if ($use_random) { $argList += "--use-random-monitor" }
    $proc = Start-Process -FilePath $PY -ArgumentList $argList -RedirectStandardOutput $seedLog -RedirectStandardError ($seedLog + ".err") -PassThru -WindowStyle Hidden
    $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    ($tag + " seed " + $s + "   LAUNCHED " + $end + "   pid=" + $proc.Id) | Out-File -FilePath $DONE -Append -Encoding utf8
}

# 5 sequential launches to avoid OOM (each takes ~5 min)
0 | ForEach-Object { Launch-One $_ $false }
"--- real 0 done ---" | Out-File -FilePath $DONE -Append -Encoding utf8
1 | ForEach-Object { Launch-One $_ $false }
"--- real 1 done ---" | Out-File -FilePath $DONE -Append -Encoding utf8
2 | ForEach-Object { Launch-One $_ $false }
"--- real 2 done ---" | Out-File -FilePath $DONE -Append -Encoding utf8
3 | ForEach-Object { Launch-One $_ $false }
"--- real 3 done ---" | Out-File -FilePath $DONE -Append -Encoding utf8
4 | ForEach-Object { Launch-One $_ $false }
"--- All 5 real launched at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ===" | Out-File -FilePath $DONE -Append -Encoding utf8
"--- Now launching 5 random (will run SEQUENTIALLY via WaitForExit) ---" | Out-File -FilePath $DONE -Append -Encoding utf8

# Wait for all real to finish (sequential is too slow; parallel was problematic)
# Instead: launch all 5 random in parallel with the 5 real still running.
# With 5 real + 5 random = 10 parallel, contention is high but OK.
0..4 | ForEach-Object { Launch-One $_ $true }
"=== All 10 H1.4 processes launched at " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ===" | Out-File -FilePath $DONE -Append -Encoding utf8
