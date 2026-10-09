@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI TW Stock Research Scheduler Status v10.0
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$names=@('AI_TW_Stock_Finance_Update_0810','AI_TW_Stock_Finance_Update_1100','AI_TW_Stock_Finance_Update_1330','AI_TW_Stock_Finance_Update_1600','AI_TW_Stock_Finance_Update_1800','AI_TW_Stock_Finance_Update_2300','AI_TW_Stock_Morning_Report','AI_TW_Stock_AfterClose_Research'); foreach($name in $names){ Write-Host ('Task        : '+$name); $task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue; if($null -eq $task){Write-Host 'State       : NOT FOUND';Write-Host '';continue}; $info=Get-ScheduledTaskInfo -TaskName $name; $r=[int64]$info.LastTaskResult; $hex=('0x{0:X8}' -f ($r -band 0xffffffff)); $desc=switch($r){0{'SUCCESS'};267011{'NEVER_RUN (0x41303)'};267009{'RUNNING (0x41301)'};267014{'TERMINATED (0x41306)'};1{'CHILD_OR_SCRIPT_FAILURE (exit 1)'};2{'FILE_NOT_FOUND (2)'};3{'PATH_NOT_FOUND (3)'};4{'SCRIPT_VALIDATION_OR_CHILD_EXIT_4'};5{'ACCESS_DENIED (5)'};6{'OUTPUT_VALIDATION_FAILED (exit 6)'};7{'GITHUB_PUBLISH_FAILED (exit 7)'};default{'RESULT '+$r}}; $stateText=$task.State; if($info.LastRunTime -lt [datetime]'2000-01-01'){ $last='NEVER'} else {$last=$info.LastRunTime}; Write-Host ('State       : '+$stateText); Write-Host ('LastRunTime : '+$last); Write-Host ('NextRunTime : '+$info.NextRunTime); Write-Host ('LastResult  : '+$desc); Write-Host ('ResultHex   : '+$hex); Write-Host ('Action      : '+$task.Actions.Execute); Write-Host ('Arguments   : '+$task.Actions.Arguments); Write-Host ('WorkingDir  : '+$task.Actions.WorkingDirectory); $trigger=$task.Triggers|Select-Object -First 1;if($null-ne$trigger){Write-Host ('Trigger     : '+$trigger.StartBoundary)};Write-Host ''}"

echo Logs:
echo %~dp0output\scheduler_logs
echo.
echo For a foreground test, run:
echo   python scheduled_job_runner.py morning
echo   python scheduled_job_runner.py afterclose
echo   python scheduled_job_runner.py finance
echo.
pause
