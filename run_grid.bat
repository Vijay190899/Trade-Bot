@echo off
setlocal EnableDelayedExpansion

set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG=%BOT_ROOT%\config\config_grid.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "LOGFILE=%USERDATA%\logs\grid_run.log"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Antigravity — GRID / DCA Bot (paper trading)
echo  Exchange : Binance  ^|  Pair: ETH/USDT  ^|  Wallet: 100 USDT
echo  Strategy : 3-mode DCA (ranging / trend / squeeze)
echo  Stake    : 25 USDT initial + 2x DCA levels = 75 USDT max
echo  Timeframe: 15m  ^|  No GPU / no training needed
echo ============================================================
echo.
echo  Modes:
echo    1. Ranging Mean Reversion  (ADX ^< 20)
echo    2. Trend Pullback          (ADX 20-40)
echo    3. Squeeze Breakout        (BB compression release)
echo.
echo  DCA Grid:
echo    L1 entry: 25 USDT
echo    L2 at -1.5%%: +25 USDT  (avg entry improves)
echo    L3 at -3.0%%: +25 USDT  (final safety net)
echo.
echo  Log: %LOGFILE%
echo  Press CTRL+C to stop.
echo.

freqtrade trade ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityGridStrategy ^
    --dry-run ^
    --logfile "%LOGFILE%"

if errorlevel 1 (
    echo.
    echo  ERROR: Grid bot failed. Check %LOGFILE% for details.
    echo.
    pause
    exit /b 1
)
