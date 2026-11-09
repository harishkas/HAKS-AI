# PowerShell smoke test launch script for HAKS AI
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "[ERROR] Virtual environment not found at $PythonExe. Please run setup_haks.ps1 first."
    exit 1
}

& $PythonExe (Join-Path $ScriptDir "app.py") --smoke-test @args
exit $LASTEXITCODE
