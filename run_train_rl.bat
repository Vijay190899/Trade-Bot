@echo off
setlocal EnableDelayedExpansion

set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "CONFIG_BT=%BOT_ROOT%\config\config_backtest.json"
set "CONFIG_RL=%BOT_ROOT%\config\config_rl.json"
set "USERDATA=%BOT_ROOT%\user_data"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"

call "%VENV%\Scripts\activate.bat"

echo ============================================================
echo  Antigravity RL Training — PPO on RTX 2070
echo  Data   : Binance ETH/USDT + BTC/USDT correlation
echo  Model  : PPO (Stable-Baselines3)  device=cuda
echo  Config : config_backtest.json + config_rl.json overlay
echo  Cycles : 25 train_cycles per window
echo ============================================================
echo.
echo  First run: ~30-60 min (GPU ~60-80%% utilisation)
echo  Subsequent runs: ~5-10 min (incremental retraining)
echo.

freqtrade backtesting ^
    --config "%CONFIG_BT%" ^
    --config "%CONFIG_RL%" ^
    --userdir "%USERDATA%" ^
    --strategy AntigravityStrategy ^
    --timerange 20250101-20260501 ^
    --enable-protections

if errorlevel 1 (
    echo.
    echo  RL training failed. Common causes:
    echo    - stable-baselines3 not installed: run setup.bat
    echo    - gymnasium not installed: run setup.bat
    echo    - CUDA out of memory: reduce train_cycles in config_rl.json
    echo    - Missing Binance data: run run_download_binance.bat first
    echo.
    pause
    exit /b 1
)

echo.
echo  RL model saved to: %USERDATA%\models\antigravity_rl_v1\
echo.
echo  Next step: run_dryrun.bat to paper trade on Kraken with the RL model
echo.
pause
