@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI TW Stock Scheduled Task Test v9.6
echo ============================================================
echo.
choice /c MAF /m "Test Morning, AfterClose, or All"
if errorlevel 3 goto all
if errorlevel 2 goto afterclose
if errorlevel 1 goto morning
exit /b 1

:morning
call :start_and_wait "AI_TW_Stock_Morning_Report"
goto end

afterclose
call :start_and_wait "AI_TW_Stock_AfterClose_Research"
goto end

:all
call :start_and_wait "AI_TW_Stock_CNYES_Nightly_News"
call :start_and_wait "AI_TW_Stock_Morning_Report"
call :start_and_wait "AI_TW_Stock_AfterClose_Research"
goto end

:start_and_wait
set "TASK=%~1"
echo.
echo ------------------------------------------------------------
echo Starting: %TASK%
echo ------------------------------------------------------------
schtasks /Run /TN "%TASK%"
if errorlevel 1 (
  echo Start request FAILED.
  exit /b 1
)

timeout /t 3 /nobreak >nul
for /l %%N in (1,1,12) do (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$t=Get-ScheduledTask -TaskName '%TASK%' -ErrorAction Stop; $i=Get-ScheduledTaskInfo -TaskName '%TASK%'; Write-Host ('State='+$t.State+' LastRun='+$i.LastRunTime+' LastResult='+$i.LastTaskResult); if($t.State -eq 'Running'){exit 10}; if($i.LastRunTime -gt (Get-Date).AddMinutes(-2)){exit 0}; exit 20"
  if not errorlevel 10 if not errorlevel 20 goto test_done
  timeout /t 2 /nobreak >nul
)

test_done
echo.
echo Latest log files:
if exist "%~dp0output\scheduler_logs" powershell.exe -NoProfile -Command "Get-ChildItem '%~dp0output\scheduler_logs' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 6 FullName,LastWriteTime,Length | Format-Table -AutoSize"
echo.
exit /b 0

:end
echo.
echo Test completed.
echo.
pause
