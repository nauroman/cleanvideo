$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv first: https://docs.astral.sh/uv/"
}

if (!(Test-Path ".venv\Scripts\python.exe")) {
    uv python install 3.10
    uv venv --python 3.10 .venv
    .\.venv\Scripts\python.exe -m ensurepip --upgrade
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt

New-Item -ItemType Directory -Force external | Out-Null
if (!(Test-Path "external\HYPIR\.git")) {
    git clone https://github.com/XPixelGroup/HYPIR.git external\HYPIR
}
if (!(Test-Path "external\SUPIR\.git")) {
    git clone https://github.com/Fanghua-Yu/SUPIR.git external\SUPIR
}

.\.venv\Scripts\python.exe scripts\download_models.py

Write-Host "CleanVideo setup complete."

