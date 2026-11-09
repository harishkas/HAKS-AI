@echo off
setlocal enabledelayedexpansion

REM Ensure working directory is the project root
cd /d "%~dp0"

echo =====================================================================
echo  HAKS AI Assistant - Environment Preparation ^& Dependency Setup
echo =====================================================================

REM 1. Inspect existing virtual environment
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Existing virtual environment detected at .venv.
    echo [INFO] Preserving existing working virtual environment - no recreation needed.
) else (
    echo [INFO] No virtual environment found at .venv. Searching for Python interpreter...
    
    set PYTHON_CMD=
    REM Check py launcher first
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        py -3 --version >nul 2>&1
        if !errorlevel! equ 0 (
            set PYTHON_CMD=py -3
        )
    )

    REM Check python on PATH
    if "!PYTHON_CMD!"=="" (
        where python >nul 2>&1
        if !errorlevel! equ 0 (
            set PYTHON_CMD=python
        )
    )

    if "!PYTHON_CMD!"=="" (
        echo [ERROR] No Python installation found on system PATH.
        echo [ERROR] Please install Python 3.10+ from https://www.python.org and add it to PATH.
        exit /b 1
    )

    echo [INFO] Creating new virtual environment at .venv using !PYTHON_CMD!...
    !PYTHON_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment in .venv.
        exit /b 1
    )
    echo [INFO] Virtual environment successfully created at .venv.
)

REM 2. Verify virtual environment interpreter
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python interpreter not found at .venv\Scripts\python.exe!
    exit /b 1
)

echo [INFO] Active project interpreter:
.venv\Scripts\python.exe --version

REM 3. Install missing dependencies into .venv
echo [INFO] Verifying and installing missing project dependencies into .venv...
.venv\Scripts\python.exe -m pip install --no-warn-script-location -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install project dependencies into .venv.
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] HAKS virtual environment is configured and ready.
echo [INFO] Interpreter: .venv\Scripts\python.exe
echo [INFO] Run app:        run_haks.bat
echo [INFO] Run smoke test: run_haks_smoke_test.bat
echo =====================================================================
exit /b 0
