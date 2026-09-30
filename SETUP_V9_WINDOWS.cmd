@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI TW Stock Research Assistant - Scheduler Setup v9.2
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_research_task.ps1"
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
    echo Scheduler setup FAILED. Error code: %ERR%
    echo.
    pause
    exit /b %ERR%
)

echo Scheduler setup completed successfully.
echo.
pause
exit /b 0
