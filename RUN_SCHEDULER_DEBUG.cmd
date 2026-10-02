@echo off
setlocal
cd /d "%~dp0"
:menu
cls
echo ============================================================
echo AI TW Stock Research Assistant - Foreground Debug Runner
echo ============================================================
echo.
echo [1] CNYES nightly news
echo [2] Morning report
echo [3] After-close research
echo [4] Show latest logs
echo [0] Exit
echo.
set /p choice=Choose: 
if "%choice%"=="1" goto cnyes
if "%choice%"=="2" goto morning
if "%choice%"=="3" goto afterclose
if "%choice%"=="4" goto logs
if "%choice%"=="0" exit /b 0
goto menu

:cnyes
python "%~dp0scheduled_job_runner.py" cnyes
echo.
pause
goto menu

:morning
python "%~dp0scheduled_job_runner.py" morning
echo.
pause
goto menu

:afterclose
python "%~dp0scheduled_job_runner.py" afterclose
echo.
pause
goto menu

:logs
powershell.exe -NoProfile -Command "Get-ChildItem '%~dp0output\scheduler_logs' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 12 FullName,LastWriteTime,Length | Format-Table -AutoSize"
echo.
pause
goto menu
