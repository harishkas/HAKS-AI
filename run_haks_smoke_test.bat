@echo off
setlocal enabledelayedexpansion

REM Ensure working directory is the project root
cd /d "%~dp0"

REM Check whether .venv\Scripts\python.exe exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo [ERROR] Please run setup_haks.bat first to initialize the environment and install dependencies.
    exit /b 1
)

.venv\Scripts\python.exe app.py --smoke-test %*
exit /b %ERRORLEVEL%
