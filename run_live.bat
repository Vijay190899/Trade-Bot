@echo off
setlocal EnableDelayedExpansion

:: ----------------------------------------------------------------
:: All paths on V: — no C: drive usage
:: ----------------------------------------------------------------
set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG=%BOT_ROOT%\config\config.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "LOGFILE=%USERDATA%\logs\live.log"
set "ENV_FILE=%BOT_ROOT%\.env"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

:: ----------------------------------------------------------------
:: Load API keys from .env (never hardcoded)
:: ----------------------------------------------------------------
if not exist "%ENV_FILE%" (
    echo.
    echo  ERROR: .env file not found at %ENV_FILE%
    echo.
    echo  Steps to fix:
    echo    1. Copy .env.example to .env
    echo    2. Fill in your Kraken API key and secret
    echo    3. Fill in your Telegram bot token and chat ID
    echo    4. Run this script again
    echo.
    pause
    exit /b 1
)

echo  Loading secrets from .env ...
for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%A"=="" if not "%%A:~0,1%"=="#" (
        set "%%A=%%B"
    )
)

:: ----------------------------------------------------------------
:: LIVE TRADING SAFETY CONFIRMATION
:: ----------------------------------------------------------------
echo.
echo ============================================================
echo  *** LIVE TRADING MODE — REAL MONEY ***
echo ============================================================
echo.
echo  Exchange  : Kraken (EU-regulated)
echo  Pair      : BTC/EUR
echo  Stake     : 9 EUR per trade (max 2 trades = 18 EUR)
echo  Stop-loss : -5%% hard stop per trade
echo  API key   : trade-only (no withdrawal permission)
echo.
echo  YOUR FUNDS ARE SAFE FROM AUTO-WITHDRAWAL.
echo  The bot can only place buy/sell orders.
echo  To withdraw: log into kraken.com manually.
echo.
echo  Have you:
echo    [Y] Completed at least 2 weeks of dry-run successfully?
echo    [Y] Reviewed the backtest results (profit factor ^> 1.5)?
echo    [Y] Set a trade-only API key (no withdrawal permission)?
echo    [Y] Deposited EUR 20 via SEPA on Kraken?
echo.
set /p CONFIRM="Type YES to start live trading (anything else aborts): "
if /i not "%CONFIRM%"=="YES" (
    echo.
    echo  Aborted. No live trading started.
    echo.
    pause
    exit /b 0
)

echo.
echo  Starting live bot... Press CTRL+C to stop safely.
echo  Web UI: http://127.0.0.1:8080
echo.

call "%VENV%\Scripts\activate.bat"

freqtrade trade ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityStrategy ^
    --logfile "%LOGFILE%"
