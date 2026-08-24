@echo off
setlocal EnableDelayedExpansion

:: ----------------------------------------------------------------
:: All paths on V: — no C: drive usage
:: ----------------------------------------------------------------
set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG=%BOT_ROOT%\config\config.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Downloading BTC/EUR historical data from Kraken
echo  Timeframes: 1h and 4h  ^|  Days: 730 (2 years)
echo ============================================================
echo.

echo  Note: Kraken requires --dl-trades (no klines API). ~15 min per 30 days.
freqtrade download-data ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --exchange kraken ^
    --pairs BTC/EUR ^
    --timeframes 1h 4h ^
    --dl-trades ^
    --days 730

if errorlevel 1 (
    echo.
    echo  ERROR: Download failed. Check your internet connection.
    echo  Note: No API key is needed for public market data.
    pause
    exit /b 1
)

echo.
echo  Data saved to: %USERDATA%\data\kraken\
echo  Ready to backtest. Run run_backtest.bat next.
echo.
pause
