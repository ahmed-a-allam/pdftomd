@echo off
REM ============================================================
REM  run.bat  -  self-building launcher (Portable)
REM  Builds .venv on first run, then starts INSTANTLY afterwards.
REM  Move this folder anywhere + double-click = it works.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.deps_ok"
set "CUDA_INDEX=https://download.pytorch.org/whl/cu128"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
REM Quieter, faster startup for the ML libraries
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TRANSFORMERS_NO_ADVISORY_WARNINGS=1"

REM --- No venv at all? Full build. ---
if not exist "%PY%" goto build

REM --- Deps already verified once? Skip ALL import checks and just run. ---
REM  (Importing torch/marker/transformers to "check" them costs 20-60s on every
REM   launch. The stamp file records that the check already passed, so normal
REM   startup is instant. Delete .venv\.deps_ok to force a re-check.)
if exist "%STAMP%" goto quran

echo [check] Verifying dependencies (one time only)...
"%PY%" -c "import torch" 1>nul 2>nul
if errorlevel 1 (
    echo [check] torch is broken - rebuilding environment.
    goto build
)
"%PY%" -c "import fitz, customtkinter, rapidfuzz, transformers, peft" 1>nul 2>nul
if errorlevel 1 (
    echo [check] Installing missing dependencies...
    "%PY%" -m pip install -r requirements.txt
)
echo ok> "%STAMP%"
goto quran

:build
echo ============================================================
echo [setup] Building the virtual environment (may take minutes)...
echo ============================================================
if exist "%VENV%" (
    echo [setup] Removing old environment...
    rmdir /s /q "%VENV%"
)

where python >nul 2>nul
if errorlevel 1 (
    echo [error] Python is not installed or not on PATH.
    echo         Get it from https://www.python.org then retry.
    pause
    exit /b 1
)

python -m venv "%VENV%"
"%PY%" -m pip install --upgrade pip
echo [setup] Installing PyTorch (GPU / CUDA 12.8)...
"%PY%" -m pip install torch torchvision --index-url %CUDA_INDEX%
echo [setup] Installing remaining dependencies...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [error] Installation failed. See messages above.
    pause
    exit /b 1
)
echo ok> "%STAMP%"
echo [setup] Build complete.

:quran
REM --- Download the Uthmani Quran corpus once (for the correction layer) ---
if not exist "data\quran-uthmani.json" (
    echo [setup] Downloading Uthmani Quran corpus...
    "%PY%" setup_quran.py
)

:run
echo [run] Launching...
"%PY%" gui.py
if errorlevel 1 pause
endlocal
