@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI TW Stock Research Scheduler Status
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"$names=@('AI_TW_Stock_CNYES_Nightly_News','AI_TW_Stock_Morning_Report','AI_TW_Stock_AfterClose_Research'); foreach($n in $names){ try { $t=Get-ScheduledTask -TaskName $n -ErrorAction Stop; $i=Get-ScheduledTaskInfo -TaskName $n -ErrorAction Stop; Write-Host ('Task        : ' + $n); Write-Host ('State       : ' + $t.State); Write-Host ('LastRunTime : ' + $i.LastRunTime); Write-Host ('NextRunTime : ' + $i.NextRunTime); Write-Host ('LastResult  : ' + $i.LastTaskResult); Write-Host '' } catch { Write-Host ('Task        : ' + $n); Write-Host 'State       : NOT FOUND' -ForegroundColor Red; Write-Host ('Error       : ' + $_.Exception.Message); Write-Host '' } }"

echo.
echo Expected schedule:
echo 08:30  Morning report
echo 14:30  After-close report
echo 23:00  CNYES nightly crawler
echo.
pause
exit /b 0
