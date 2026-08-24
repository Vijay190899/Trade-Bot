@echo off
setlocal
set "VENV=V:\Antigravity\Trade tool\bot\venv\Scripts\python.exe"
set "DASH=V:\Antigravity\Trade tool\bot\ui\dashboard.py"

:retry
"%VENV%" "%DASH%"
if errorlevel 1 (
    timeout /t 10 /nobreak >nul
    goto retry
)
