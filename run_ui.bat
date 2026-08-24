@echo off
setlocal EnableDelayedExpansion

set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "DASHBOARD=%BOT_ROOT%\ui\dashboard.py"
set "PORT=8899"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Antigravity — Monitoring Dashboard
echo  URL  : http://localhost:%PORT%
echo  Data : Live Binance price + dry-run SQLite + TradeMemory
echo  Auto-refresh every 5 seconds
echo ============================================================
echo.
echo  Open your browser at http://localhost:%PORT%
echo  Press CTRL+C to stop the dashboard.
echo.

python "%DASHBOARD%"

if errorlevel 1 (
    echo.
    echo  ERROR: Dashboard failed to start. Check console output above.
    echo.
    pause
    exit /b 1
)
