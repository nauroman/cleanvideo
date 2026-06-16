param(
    [switch]$NoBrowser,
    [switch]$Restart,
    [switch]$RestartWhenIdle,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Url = "http://127.0.0.1:$Port"
$OpenUrl = "$Url/?v=2026-06-16-resource-mode-v20"
$HealthUrl = "$Url/api/health"
$StatusUrl = "$Url/api/status"

function Get-CleanVideoStatus {
    try {
        return Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 5
    } catch {
        try {
            return Invoke-RestMethod -Uri $StatusUrl -TimeoutSec 15
        } catch {
            return $null
        }
    }
}

function Test-PortListening {
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

function Get-PortProcessIds {
    try {
        return @(
            Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique
        )
    } catch {
        return @()
    }
}

function Get-PortProcesses {
    $processes = @()
    foreach ($processId in Get-PortProcessIds) {
        if ($processId -gt 0) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
            if ($null -ne $process) {
                $processes += $process
            }
        }
    }
    return $processes
}

function Test-PortOwnedByCleanVideo {
    foreach ($process in Get-PortProcesses) {
        if (($process.CommandLine -match "uvicorn\s+app\.main:app") -or ($process.ExecutablePath -like "$Root*")) {
            return $true
        }
    }
    return $false
}

function Get-PortOwnerSummary {
    $owners = @()
    foreach ($process in Get-PortProcesses) {
        $commandLine = if ($process.CommandLine) { $process.CommandLine } else { $process.ExecutablePath }
        $owners += "PID $($process.ProcessId): $commandLine"
    }
    if ($owners.Count -eq 0) {
        return "unknown owner"
    }
    return ($owners -join "; ")
}

function Stop-PortProcesses {
    foreach ($processId in Get-PortProcessIds) {
        if ($processId -gt 0) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }

    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if (!(Test-PortListening)) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
}

if (!(Test-Path $Python)) {
    throw "Missing virtual environment. Run scripts\setup.ps1 first. Expected: $Python"
}

$status = Get-CleanVideoStatus
$isCleanVideo = $null -ne $status -and $status.app -eq "CleanVideo"

if ($isCleanVideo -and ($Restart -or $RestartWhenIdle)) {
    $activeJobs = 0
    if ($null -ne $status.activeJobs) {
        $activeJobs = [int]$status.activeJobs
    }

    if ($Restart -or $activeJobs -eq 0) {
        Write-Host "Restarting CleanVideo on port $Port..."
        Stop-PortProcesses
        $status = $null
        $isCleanVideo = $false
    } else {
        Write-Host "CleanVideo has $activeJobs active job(s); reusing the running server."
    }
}

if (!$isCleanVideo) {
    if (Test-PortListening) {
        if (($Restart -or $RestartWhenIdle) -and (Test-PortOwnedByCleanVideo)) {
            Write-Host "Port $Port is held by a stale CleanVideo process; stopping it before restart..."
            Stop-PortProcesses
        } else {
            $owners = Get-PortOwnerSummary
            throw "Port $Port is already in use, but CleanVideo is not responding at $HealthUrl or $StatusUrl. Owner: $owners"
        }
    }

    if (Test-PortListening) {
        $owners = Get-PortOwnerSummary
        throw "Port $Port is still in use after cleanup. Owner: $owners"
    }

    Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $status = Get-CleanVideoStatus
        if ($null -ne $status -and $status.app -eq "CleanVideo") {
            $isCleanVideo = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
}

if (!$isCleanVideo) {
    throw "CleanVideo did not become ready at $StatusUrl."
}

if (!$NoBrowser) {
    Start-Process $OpenUrl
}

Write-Host "CleanVideo is running at $Url"
