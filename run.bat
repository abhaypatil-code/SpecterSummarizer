@echo off
REM ============================================================================
REM  SpecterSummarizer - one-click launcher
REM  Double-click this file to set up (if needed) and open the app.
REM ============================================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   SpecterSummarizer
echo ============================================================
echo.

REM --- Find Python ------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH. Install Python 3.10+ first.
    pause
    exit /b 1
)

REM --- Create / activate a virtual environment --------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
)
set "PY=.venv\Scripts\python.exe"

REM --- Install dependencies (only if Streamlit is missing) --------------------
"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies ^(first run only, may take a few minutes^)...
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

REM --- Build data splits if missing -------------------------------------------
if not exist "data\test_processed.jsonl" (
    echo [setup] Building data splits...
    "%PY%" -m src.preprocess
)

REM --- Train a quick demo model if none exists --------------------------------
if not exist "models\t5_summarizer\config.json" (
    echo [setup] No fine-tuned model found. Training a quick demo model...
    echo         ^(For full quality run: python -m src.train --epochs 3 --max_samples 0^)
    "%PY%" -m src.train --epochs 1 --max_samples 100
)

REM --- Launch the app ---------------------------------------------------------
echo.
echo [run] Starting SpecterSummarizer... your browser will open shortly.
echo       Press Ctrl+C in this window to stop the app.
echo.
"%PY%" -m streamlit run app.py

pause
endlocal
