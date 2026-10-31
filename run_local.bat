@echo off
cd /d "%~dp0"
if not exist ".venv-local\Scripts\python.exe" (
    echo Python environment not found. Run setup_local.ps1 first.
    pause
    exit /b 1
)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false
set PYTHONUTF8=1
".venv-local\Scripts\python.exe" launch_local.pyw
if errorlevel 1 pause
