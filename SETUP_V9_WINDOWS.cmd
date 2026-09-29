@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo AI TW Stock Research Assistant v9.1 - Windows Scheduler Setup
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_research_task.ps1"
echo.
echo Scheduler setup finished.
pause
