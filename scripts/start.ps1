param(
    [switch]$NoBrowser,
    [switch]$Restart,
    [switch]$RestartWhenIdle,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

& (Join-Path $Root "scripts\setup.ps1") -IfNeeded

$launchArgs = @{
    Port = $Port
}

if ($NoBrowser) {
    $launchArgs.NoBrowser = $true
}

if ($Restart) {
    $launchArgs.Restart = $true
}

if ($RestartWhenIdle) {
    $launchArgs.RestartWhenIdle = $true
}

& (Join-Path $Root "scripts\launch.ps1") @launchArgs
