@echo off
setlocal EnableDelayedExpansion

set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG_BT=%BOT_ROOT%\config\config_backtest.json"
set "CONFIG_RL=%BOT_ROOT%\config\config_rl.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "LOGFILE=%USERDATA%\logs\dry_run.log"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Antigravity — DRY RUN (paper trading, no real money)
echo  Exchange : Binance  ^|  Pair: ETH/USDT  ^|  Wallet: 100 USDT
echo  Model    : PPO RL (AntigravityRLModel) — self-improving
echo  Config   : config_backtest.json + config_rl.json overlay
echo ============================================================
echo.
echo  No API key required — Binance public data only.
echo  Simulated fills at live market prices.
echo  TradeMemory will log real-time outcomes to improve the model.
echo.
echo  Retrain interval : every 8 hours (live_retrain_hours)
echo  Log file         : %LOGFILE%
echo.
echo  Press CTRL+C to stop.
echo.

freqtrade trade ^
    --config "%CONFIG_BT%" ^
    --config "%CONFIG_RL%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityStrategy ^
    --dry-run ^
    --logfile "%LOGFILE%"

if errorlevel 1 (
    echo.
    echo  ERROR: Dry run failed. Check %LOGFILE% for details.
    echo.
    pause
    exit /b 1
)
