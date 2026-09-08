@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found: venv
    echo Create it and install requirements before running the dashboard.
    pause
    exit /b 1
)
if not exist "social_radar.duckdb" (
    echo No database found. Run run.bat first to collect and analyze data.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 pause
endlocal
