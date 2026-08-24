@echo off
setlocal EnableDelayedExpansion

set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG=%BOT_ROOT%\config\config_grid.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Antigravity — Grid/DCA Hyperopt
echo  Pair     : ETH/USDT  ^|  Timeframe: 15m
echo  Spaces   : buy (RSI ranging, BB pos, RSI trend) + sell (RSI exit, BB exit)
echo  Loss fn  : SortinoHyperOptLoss (downside-risk aware)
echo  Epochs   : 300
echo  Timerange: 2024-06-01 to today (~12 months of 15m data)
echo ============================================================
echo.

echo [1/2] Downloading 15m ETH/USDT data (2024-06-01 to today)...
echo.

freqtrade download-data ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --timeframe 15m ^
    --timerange 20240601- ^
    --exchange binance

if errorlevel 1 (
    echo.
    echo  ERROR: Data download failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2/2] Running hyperopt (300 epochs, ~10-20 minutes)...
echo  Optimising: buy_rsi_ranging, buy_bb_pos, buy_rsi_trend, sell_rsi_exit, sell_bb_exit
echo.

freqtrade hyperopt ^
    --config "%CONFIG%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityGridStrategy ^
    --hyperopt-loss SortinoHyperOptLoss ^
    --spaces buy sell ^
    --epochs 300 ^
    --timerange 20240601- ^
    --min-trades 30

if errorlevel 1 (
    echo.
    echo  ERROR: Hyperopt failed. See output above for details.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Hyperopt complete.
echo  Best params are printed above.
echo  To apply them, update AntigravityGridStrategy.py defaults:
echo    buy_rsi_ranging  = IntParameter(..., default=XX)
echo    buy_bb_pos       = DecimalParameter(..., default=0.XX)
echo    buy_rsi_trend    = IntParameter(..., default=XX)
echo    sell_rsi_exit    = IntParameter(..., default=XX)
echo    sell_bb_exit     = DecimalParameter(..., default=0.XX)
echo ============================================================
echo.
pause
