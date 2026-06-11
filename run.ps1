$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Missing virtual environment. Expected: $Python"
}
Set-Location $Root
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

