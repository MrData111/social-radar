@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found: venv
    echo Create it and install requirements before running the pipeline.
    pause
    exit /b 1
)
echo Starting data collection and AI analysis...
echo The program will ask for search settings in the console.
echo.
"venv\Scripts\python.exe" run_pipeline.py
if errorlevel 1 pause
endlocal
