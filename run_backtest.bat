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
echo  Running backtest — AntigravityStrategy on BTC/EUR
echo  Period: last 365 days  ^|  FreqAI enabled
echo ============================================================
echo.
echo  This will train the LightGBM model on your RTX 2070.
echo  First run takes ~10-15 minutes. Subsequent runs are faster.
echo.

freqtrade backtesting ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityStrategy ^
    --timerange 20240101- ^
    --enable-protections ^
    --freqai-backtest-live-models 0

if errorlevel 1 (
    echo.
    echo  ERROR: Backtest failed. Check the output above.
    echo  Common issues:
    echo    - Not enough data: Run run_download.bat first
    echo    - Missing dependencies: Run setup.bat again
    pause
    exit /b 1
)

echo.
echo  Results saved to: %USERDATA%\backtest_results\
echo.
echo  Key metrics to check:
echo    - Profit factor ^> 1.5
echo    - Sharpe ratio  ^> 1.0
echo    - Max drawdown  ^< 15%%
echo    - Win rate      ^> 45%%
echo    - Avg profit    ^> 0.5%% (above Kraken fees)
echo.
pause
