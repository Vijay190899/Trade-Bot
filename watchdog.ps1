# Qoherenz Bot Watchdog
# Freqtrade sometimes hangs as a zombie process after a network error during
# startup (reload_markets timeout) instead of exiting cleanly. Windows still
# sees the PID as alive ("Responding: True", ~0 CPU) so Task Scheduler's
# restart-on-failure never triggers. This watchdog checks the actual log
# heartbeat freshness (freqtrade logs one every ~60s) and force-restarts
# any bot whose log has gone stale, regardless of whether the process
# object still exists.

$botRoot = "V:\Antigravity\Trade tool\bot"
$venv    = "$botRoot\venv\Scripts"
$staleMinutes = 5

function Get-LastHeartbeat($logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $line = Get-Content $logPath -Tail 40 | Where-Object { $_ -match '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' } | Select-Object -Last 1
    if (-not $line) { return $null }
    $ts = $line.Substring(0,19)
    try { return [datetime]::ParseExact($ts, "yyyy-MM-dd HH:mm:ss", $null) } catch { return $null }
}

function Restart-Bot($name, $exe, $args, $logPath, $matchStr) {
    Get-CimInstance Win32_Process -Filter "Name='freqtrade.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $matchStr -and $_.CommandLine -notmatch 'backtesting' } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Output "$(Get-Date -Format s)  [$name] killed stale PID $($_.ProcessId)"
        }
    Start-Sleep -Seconds 2
    $p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $botRoot -WindowStyle Hidden -PassThru
    Write-Output "$(Get-Date -Format s)  [$name] relaunched PID $($p.Id)"
}

# ---- RL Bot ----
$rlLog = "$botRoot\user_data\logs\dry_run.log"
$rlHb  = Get-LastHeartbeat $rlLog
$rlProc = Get-CimInstance Win32_Process -Filter "Name='freqtrade.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'trade' -and $_.CommandLine -match 'AntigravityStrategy(?!V)' -and $_.CommandLine -notmatch 'Grid|backtesting' }

if (-not $rlProc) {
    Write-Output "$(Get-Date -Format s)  [RL] not running - starting"
    Restart-Bot "RL" "$venv\freqtrade.exe" @("trade","--config","`"$botRoot\config\config_backtest.json`"","--config","`"$botRoot\config\config_rl.json`"","--userdir","`"$botRoot\user_data`"","--strategy","AntigravityStrategy","--dry-run","--logfile","`"$rlLog`"") $rlLog 'AntigravityStrategy'
} elseif ($rlHb -eq $null -or ((Get-Date) - $rlHb).TotalMinutes -gt $staleMinutes) {
    Write-Output "$(Get-Date -Format s)  [RL] heartbeat stale (last: $rlHb) - restarting"
    Restart-Bot "RL" "$venv\freqtrade.exe" @("trade","--config","`"$botRoot\config\config_backtest.json`"","--config","`"$botRoot\config\config_rl.json`"","--userdir","`"$botRoot\user_data`"","--strategy","AntigravityStrategy","--dry-run","--logfile","`"$rlLog`"") $rlLog 'AntigravityStrategy'
}

# ---- Grid Bot ----
$gridLog = "$botRoot\user_data\logs\grid_run.log"
$gridHb  = Get-LastHeartbeat $gridLog
$gridProc = Get-CimInstance Win32_Process -Filter "Name='freqtrade.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'trade' -and $_.CommandLine -match 'AntigravityGridStrategy' -and $_.CommandLine -notmatch 'backtesting' }

if (-not $gridProc) {
    Write-Output "$(Get-Date -Format s)  [Grid] not running - starting"
    Restart-Bot "Grid" "$venv\freqtrade.exe" @("trade","--config","`"$botRoot\config\config_grid.json`"","--userdir","`"$botRoot\user_data`"","--strategy","AntigravityGridStrategy","--dry-run","--logfile","`"$gridLog`"") $gridLog 'AntigravityGridStrategy'
} elseif ($gridHb -eq $null -or ((Get-Date) - $gridHb).TotalMinutes -gt $staleMinutes) {
    Write-Output "$(Get-Date -Format s)  [Grid] heartbeat stale (last: $gridHb) - restarting"
    Restart-Bot "Grid" "$venv\freqtrade.exe" @("trade","--config","`"$botRoot\config\config_grid.json`"","--userdir","`"$botRoot\user_data`"","--strategy","AntigravityGridStrategy","--dry-run","--logfile","`"$gridLog`"") $gridLog 'AntigravityGridStrategy'
}

# ---- Dashboard ----
# Checking "is something listening on the port" isn't enough — another app
# (e.g. a Docker container) can grab the same port and this would never
# notice. Verify the owning process is actually our dashboard.py.
$dashConn = Get-NetTCPConnection -LocalPort 8899 -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -eq '0.0.0.0' } | Select-Object -First 1
$dashOurs = $false
if ($dashConn) {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($dashConn.OwningProcess)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmd -match 'dashboard\.py') { $dashOurs = $true }
}
if (-not $dashOurs) {
    Write-Output "$(Get-Date -Format s)  [Dashboard] not running on 8899 - starting"
    Start-Process -FilePath "$venv\python.exe" -ArgumentList "`"$botRoot\ui\dashboard.py`"" -WorkingDirectory $botRoot -WindowStyle Hidden | Out-Null
}
