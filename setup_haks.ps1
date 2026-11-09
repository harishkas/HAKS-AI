# PowerShell environment setup script for HAKS AI
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " HAKS AI Assistant - Environment Preparation & Dependency Setup" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

if (Test-Path $PythonExe) {
    Write-Host "[INFO] Existing virtual environment detected at $VenvDir." -ForegroundColor Green
    Write-Host "[INFO] Preserving existing working virtual environment (no recreation)." -ForegroundColor Green
} else {
    Write-Host "[INFO] No virtual environment found. Searching for system Python..." -ForegroundColor Yellow
    $SysPython = (Get-Command py, python -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $SysPython) {
        Write-Error "[ERROR] No Python installation found on PATH. Please install Python 3.10+ from python.org."
        exit 1
    }

    Write-Host "[INFO] Creating virtual environment using $($SysPython.Source)..." -ForegroundColor Yellow
    & $SysPython.Source -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        Write-Error "[ERROR] Failed to create virtual environment."
        exit 1
    }
    Write-Host "[INFO] Virtual environment created successfully at $VenvDir." -ForegroundColor Green
}

$PyVersion = & $PythonExe --version 2>&1
Write-Host "[INFO] Active project interpreter: $PyVersion" -ForegroundColor Green

Write-Host "[INFO] Checking and installing missing project dependencies into .venv..." -ForegroundColor Cyan
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"
& $PythonExe -m pip install --no-warn-script-location -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] Failed to install project dependencies."
    exit 1
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] HAKS virtual environment is configured and ready!" -ForegroundColor Green
Write-Host "[INFO] Interpreter: $PythonExe" -ForegroundColor Green
Write-Host "[INFO] Run app:        .\run_haks.ps1 or run_haks.bat" -ForegroundColor Green
Write-Host "[INFO] Run smoke test: .\run_haks_smoke_test.ps1 or run_haks_smoke_test.bat" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
