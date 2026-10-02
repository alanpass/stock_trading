@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo AI TW Stock Research Assistant - Scheduler Setup v9.6
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_research_task_v9_6.ps1"
if errorlevel 1 (
  echo.
  echo Scheduler setup FAILED.
  pause
  exit /b 1
)
echo.
echo Scheduler setup completed.
echo Run check_research_task.cmd now.
echo.
pause
