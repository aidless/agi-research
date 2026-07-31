# run_r1_after_wait.ps1
# Waits until 01:05 then launches R1 test.
# This is a wrapper that respects the 1-hour waiting period.

$ErrorActionPreference = "Continue"
$WAIT_UNTIL = Get-Date "2026-08-01 01:05:00"
$NOW = Get-Date
$WAIT_SEC = ($WAIT_UNTIL - $NOW).TotalSeconds

if ($WAIT_SEC -gt 0) {
    Write-Host "Waiting $WAIT_SEC seconds until 01:05 to launch R1 test..."
    Start-Sleep -Seconds ([int]$WAIT_SEC)
}

Write-Host "01:05 reached, launching R1 test..."
& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "E:\agi-research\experiments_log\_run_r1_test.ps1"