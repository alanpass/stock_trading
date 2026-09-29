@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo AI TW Stock Research Scheduler Status
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$names = @('AI_TW_Stock_CNYES_Nightly_News','AI_TW_Stock_Morning_Report','AI_TW_Stock_AfterClose_Research'); foreach ($name in $names) { Write-Host ('Task        : ' + $name); $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue; if ($null -eq $task) { Write-Host 'State       : NOT FOUND' -ForegroundColor Red; Write-Host ''; continue }; $info = Get-ScheduledTaskInfo -TaskName $name; Write-Host ('State       : ' + $task.State); Write-Host ('Next run    : ' + $info.NextRunTime); Write-Host ('Last run    : ' + $info.LastRunTime); Write-Host ('Last result : ' + $info.LastTaskResult); Write-Host ('Missed runs : ' + $info.NumberOfMissedRuns); Write-Host '' }"
echo.
echo Expected schedule:
echo 08:30  Morning report
echo 14:30  After-close report
echo 23:00  CNYES nightly crawler
echo.
pause
