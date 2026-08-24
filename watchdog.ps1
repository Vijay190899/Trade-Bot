# Qoherenz Bot Watchdog
#
# Freqtrade sometimes hangs as a zombie after a network error during startup
# (reload_markets timeout) instead of exiting cleanly. Windows still sees the
# PID as alive ("Responding: True", ~0 CPU) so Task Scheduler's restart-on-
# failure never triggers. This watchdog checks actual log heartbeat freshness
# (freqtrade logs one every ~60s) and force-restarts any bot whose log has
# gone stale, regardless of whether a process object still exists.
#
# Three bugs previously made this script silently non-functional:
#   1. `$args` is a PowerShell AUTOMATIC variable and cannot be bound as a
#      parameter, so it was always empty and Start-Process always threw.
#      The watchdog never once restarted anything. Renamed to $argList.
#   2. It matched only `freqtrade.exe`, but that is a thin launcher which
#      spawns the real bot as a `python.exe` CHILD. Orphaned workers (parent
#      dead, child alive) were invisible, causing duplicate bots on one DB.
#      Now matches both image names.
#   3. Stop-Process killed the launcher but left the python child running,
#      orphaning a worker on every restart. Now kills the whole tree.
#
# Output is written to user_data/logs/watchdog.log because Task Scheduler
# discards stdout - which is how bugs 1-3 went unnoticed for 10 days.

$botRoot      = "V:\Antigravity\Trade tool\bot"
$venv         = "$botRoot\venv\Scripts"
$staleMinutes = 5
$wdLog        = "$botRoot\user_data\logs\watchdog.log"

function Write-Log($msg) {
    $line = "$(Get-Date -Format s)  $msg"
    Write-Output $line
    try { Add-Content -Path $wdLog -Value $line -ErrorAction SilentlyContinue } catch { }
}

function Get-LastHeartbeat($logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $line = Get-Content $logPath -Tail 40 -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' } |
            Select-Object -Last 1
    if (-not $line) { return $null }
    try { return [datetime]::ParseExact($line.Substring(0,19), "yyyy-MM-dd HH:mm:ss", $null) }
    catch { return $null }
}

# Match BOTH the freqtrade.exe launcher and its python.exe worker child.
# Anchor on '--strategy' rather than a bare 'trade': the install path is
# 'V:\Antigravity\Trade tool\bot', so ANY process started from this venv
# (e.g. the VS Code python language server) matches a bare 'trade'.
# Exclude backtesting runs and the V3 research strategy so neither is ever
# mistaken for the live bot (or killed by it).
function Get-BotProcs($strategyPattern, $excludePattern) {
    Get-CimInstance Win32_Process -Filter "Name='freqtrade.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match '--strategy' -and
            $_.CommandLine -match $strategyPattern -and
            $_.CommandLine -notmatch 'backtesting' -and
            ($excludePattern -eq '' -or $_.CommandLine -notmatch $excludePattern)
        }
}

function Stop-Tree($procId, $name) {
    # taskkill /T kills the whole tree; Stop-Process would orphan the child.
    & taskkill.exe /PID $procId /T /F 2>&1 | Out-Null
    Write-Log "[$name] killed PID $procId (tree)"
}

function Restart-Bot($name, $exe, $argList, $strategyPattern, $excludePattern) {
    foreach ($p in @(Get-BotProcs $strategyPattern $excludePattern)) {
        Stop-Tree $p.ProcessId $name
    }
    Start-Sleep -Seconds 3
    $p = Start-Process -FilePath $exe -ArgumentList $argList `
                       -WorkingDirectory $botRoot -WindowStyle Hidden -PassThru
    Write-Log "[$name] relaunched PID $($p.Id)"
}

# ---------------- RL Bot ----------------
$rlLog  = "$botRoot\user_data\logs\dry_run.log"
$rlArgs = @("trade",
            "--config","`"$botRoot\config\config_backtest.json`"",
            "--config","`"$botRoot\config\config_rl.json`"",
            "--userdir","`"$botRoot\user_data`"",
            "--strategy","AntigravityStrategy",
            "--dry-run",
            "--logfile","`"$rlLog`"")
$rlPat  = 'AntigravityStrategy'
$rlExcl = 'Grid|AntigravityStrategyV'

$rlProcs = @(Get-BotProcs $rlPat $rlExcl)
$rlHb    = Get-LastHeartbeat $rlLog

if ($rlProcs.Count -eq 0) {
    Write-Log "[RL] not running - starting"
    Restart-Bot "RL" "$venv\freqtrade.exe" $rlArgs $rlPat $rlExcl
}
elseif ($rlHb -eq $null -or ((Get-Date) - $rlHb).TotalMinutes -gt $staleMinutes) {
    Write-Log "[RL] heartbeat stale (last: $rlHb) - restarting"
    Restart-Bot "RL" "$venv\freqtrade.exe" $rlArgs $rlPat $rlExcl
}
elseif ($rlProcs.Count -gt 2) {
    # Expect at most 2: launcher + worker. More means duplicates on one DB.
    Write-Log "[RL] $($rlProcs.Count) processes (duplicates) - consolidating"
    Restart-Bot "RL" "$venv\freqtrade.exe" $rlArgs $rlPat $rlExcl
}

# ---------------- Grid Bot ----------------
$gridLog  = "$botRoot\user_data\logs\grid_run.log"
$gridArgs = @("trade",
              "--config","`"$botRoot\config\config_grid.json`"",
              "--userdir","`"$botRoot\user_data`"",
              "--strategy","AntigravityGridStrategy",
              "--dry-run",
              "--logfile","`"$gridLog`"")
$gPat  = 'AntigravityGridStrategy'
$gExcl = ''

$gridProcs = @(Get-BotProcs $gPat $gExcl)
$gridHb    = Get-LastHeartbeat $gridLog

if ($gridProcs.Count -eq 0) {
    Write-Log "[Grid] not running - starting"
    Restart-Bot "Grid" "$venv\freqtrade.exe" $gridArgs $gPat $gExcl
}
elseif ($gridHb -eq $null -or ((Get-Date) - $gridHb).TotalMinutes -gt $staleMinutes) {
    Write-Log "[Grid] heartbeat stale (last: $gridHb) - restarting"
    Restart-Bot "Grid" "$venv\freqtrade.exe" $gridArgs $gPat $gExcl
}
elseif ($gridProcs.Count -gt 2) {
    Write-Log "[Grid] $($gridProcs.Count) processes (duplicates) - consolidating"
    Restart-Bot "Grid" "$venv\freqtrade.exe" $gridArgs $gPat $gExcl
}

# ---------------- Dashboard ----------------
# Checking "is something listening on the port" isn't enough - another app
# (e.g. a Docker container) can grab the port and this would never notice.
# Verify the owning process is actually our dashboard.py.
$dashConn = Get-NetTCPConnection -LocalPort 8899 -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalAddress -eq '0.0.0.0' } | Select-Object -First 1
$dashOurs = $false
if ($dashConn) {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($dashConn.OwningProcess)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmd -match 'dashboard\.py') { $dashOurs = $true }
}
if (-not $dashOurs) {
    Write-Log "[Dashboard] not running on 8899 - starting"
    Start-Process -FilePath "$venv\python.exe" -ArgumentList "`"$botRoot\ui\dashboard.py`"" `
                  -WorkingDirectory $botRoot -WindowStyle Hidden | Out-Null
}
