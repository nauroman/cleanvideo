@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" -RestartWhenIdle
if errorlevel 1 (
  echo.
  echo CleanVideo failed to start. See the message above.
  pause
)
