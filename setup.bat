@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Antigravity Trading Bot — Setup v2 (RL Edition, V: drive only)
echo ============================================================
echo.

:: ----------------------------------------------------------------
:: All paths stay on V: — nothing written to C:\Users or AppData
:: ----------------------------------------------------------------
set "BOT_ROOT=V:\Antigravity\Trade tool\bot"
set "VENV=%BOT_ROOT%\venv"
set "PIP_CACHE_DIR=%BOT_ROOT%\.pip-cache"
set "TEMP=%BOT_ROOT%\.tmp"
set "TMP=%BOT_ROOT%\.tmp"
set "TMPDIR=%BOT_ROOT%\.tmp"
set "PYTHONDONTWRITEBYTECODE=1"

:: ----------------------------------------------------------------
:: Check Python 3.11
:: ----------------------------------------------------------------
echo [1/7] Checking Python version...
python --version 2>nul | findstr /C:"3.11" >nul
if errorlevel 1 (
    python --version 2>nul | findstr /C:"3.10" >nul
    if errorlevel 1 (
        echo.
        echo  ERROR: Python 3.10 or 3.11 is required.
        echo  Download from: https://www.python.org/downloads/release/python-3119/
        echo  Make sure to check "Add Python to PATH" during install.
        echo.
        pause
        exit /b 1
    )
)
echo  OK — Python found.

:: ----------------------------------------------------------------
:: Create virtual environment on V:
:: ----------------------------------------------------------------
echo.
echo [2/7] Creating virtual environment at %VENV% ...
if exist "%VENV%\Scripts\activate.bat" (
    echo  Already exists — skipping creation.
) else (
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo  ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo  Created.
)

:: ----------------------------------------------------------------
:: Activate venv
:: ----------------------------------------------------------------
call "%VENV%\Scripts\activate.bat"

:: ----------------------------------------------------------------
:: Upgrade pip + setuptools (using V: cache)
:: ----------------------------------------------------------------
echo.
echo [3/7] Upgrading pip ...
pip install --upgrade pip setuptools wheel --cache-dir "%PIP_CACHE_DIR%" --quiet
if errorlevel 1 (
    echo  ERROR: pip upgrade failed.
    pause
    exit /b 1
)
echo  Done.

:: ----------------------------------------------------------------
:: Install Freqtrade with FreqAI extras
:: ----------------------------------------------------------------
echo.
echo [4/7] Installing Freqtrade + FreqAI + RL (this takes 3-5 minutes) ...
pip install "freqtrade[freqai,freqai-rl]" --cache-dir "%PIP_CACHE_DIR%" --quiet
if errorlevel 1 (
    echo  ERROR: Freqtrade install failed.
    echo  Try running this manually:
    echo    pip install "freqtrade[freqai]" --cache-dir "%PIP_CACHE_DIR%"
    pause
    exit /b 1
)
echo  Done.

:: ----------------------------------------------------------------
:: Install PyTorch with CUDA 11.8 for RTX 2070
:: ----------------------------------------------------------------
echo.
echo [5/7] Installing PyTorch with CUDA 11.8 (RTX 2070) ...
echo  (Large download ~2 GB — please wait...)
pip install torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu118 ^
    --cache-dir "%PIP_CACHE_DIR%" ^
    --quiet
if errorlevel 1 (
    echo  WARNING: PyTorch CUDA install failed. Retrying with CPU-only...
    pip install torch torchvision torchaudio --cache-dir "%PIP_CACHE_DIR%" --quiet
)
echo  Done.

:: ----------------------------------------------------------------
:: Install LightGBM (GPU support via OpenCL — included with NVIDIA driver)
:: ----------------------------------------------------------------
echo.
echo [6/7] Installing LightGBM ...
pip install lightgbm --cache-dir "%PIP_CACHE_DIR%" --quiet
echo  Done.

:: ----------------------------------------------------------------
:: Install additional dependencies
:: ----------------------------------------------------------------
echo.
echo [7/7] Installing remaining dependencies (RL + indicators) ...
pip install ^
    pandas-ta ^
    scikit-learn ^
    optuna ^
    psutil ^
    stable-baselines3[extra] ^
    gymnasium ^
    shimmy ^
    --cache-dir "%PIP_CACHE_DIR%" ^
    --quiet
echo  Done.

:: ----------------------------------------------------------------
:: Verify installation
:: ----------------------------------------------------------------
echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
python -c "import freqtrade; print('  freqtrade    :', freqtrade.__version__)"
python -c "import torch; print('  torch        :', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import lightgbm; print('  lightgbm     :', lightgbm.__version__)"
python -c "import stable_baselines3; print('  stable-baselines3:', stable_baselines3.__version__)"
python -c "import gymnasium; print('  gymnasium    :', gymnasium.__version__)"
python -c "import pandas_ta; print('  pandas_ta    : OK')"

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo    1. Copy .env.example to .env and fill in your Kraken + Telegram keys
echo    2. run_download.bat    -- download 2 years of BTC/EUR data
echo    3. run_backtest.bat    -- backtest supervised (LightGBM) strategy
echo    4. run_train_rl.bat    -- train PPO RL agent on RTX 2070
echo    5. run_dryrun.bat      -- paper trade (no real money, 2+ weeks)
echo    6. run_live.bat        -- go live with real EUR20
echo.
pause
