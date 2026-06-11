param(
    [switch]$NoBrowser,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Url = "http://127.0.0.1:$Port"
$StatusUrl = "$Url/api/status"

function Test-CleanVideoReady {
    try {
        $response = Invoke-RestMethod -Uri $StatusUrl -TimeoutSec 2
        return $response.app -eq "CleanVideo"
    } catch {
        return $false
    }
}

function Test-PortListening {
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

if (!(Test-Path $Python)) {
    throw "Missing virtual environment. Run scripts\setup.ps1 first. Expected: $Python"
}

if (!(Test-CleanVideoReady)) {
    if (Test-PortListening) {
        throw "Port $Port is already in use, but CleanVideo is not responding at $StatusUrl."
    }

    Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $Root `
        -WindowStyle Minimized

    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if (Test-CleanVideoReady) {
            break
        }
        Start-Sleep -Milliseconds 500
    }
}

if (!(Test-CleanVideoReady)) {
    throw "CleanVideo did not become ready at $StatusUrl."
}

if (!$NoBrowser) {
    Start-Process $Url
}

Write-Host "CleanVideo is running at $Url"
