# AI TW Stock Research Scheduler v13
#
#   Finance info - NEWS ONLY (news crawl + Agent summary, earnings are NOT crawled):
#       08:10  11:00  13:30  16:00  23:00   (every day)
#   Earnings (full update = news + earnings call memos, once a day):
#       18:00   (every day)
#   08:30  Morning report  (uses the 08:10 finance cache)
#   14:30  After-close analysis report
#
# Each finance run REPLACES the previous finance_info_latest.json.
# WakeToRun ON, no overlapping runs.
# Finance: missed slots run when the PC is back (StartWhenAvailable) + a catch-up run after every logon.
# Reports (morning / after-close): never caught up late.
#
# Usage (normal PowerShell, no admin needed):
#   cd <project folder>
#   powershell -ExecutionPolicy Bypass -File .\setup_research_task_v13.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Runner = Join-Path $ProjectDir "scheduled_job_runner.py"

$PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) { $PythonCommand = Get-Command python -ErrorAction SilentlyContinue }
if ($null -eq $PythonCommand) {
    Write-Host "ERROR: python.exe not found." -ForegroundColor Red
    exit 1
}
$PythonPath = $PythonCommand.Source
if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py not found: $Runner" -ForegroundColor Red
    exit 1
}

$CurrentUser = if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) { $env:USERNAME } else { "$env:USERDOMAIN\$env:USERNAME" }

$Tasks = @(
    @{ Name = "AI_TW_Stock_Finance_Info";       Job = "finance";    Times = @("08:10", "11:00", "13:30", "16:00", "23:00") },
    @{ Name = "AI_TW_Stock_Earnings_Daily";     Job = "earnings";   Times = @("18:00") },
    @{ Name = "AI_TW_Stock_Morning_Report";     Job = "morning";    Times = @("08:30") },
    @{ Name = "AI_TW_Stock_AfterClose_Research"; Job = "afterclose"; Times = @("14:30") }
)

# Remove older task names (v9/v10/v11 and the separate nightly CNYES task).
# NOTE: an even older task named TaiwanStock_ModelResearch_Agent (15:30) is NOT touched - delete it yourself
# in Task Scheduler if it still exists, otherwise the after-close mail may be sent twice.
$OldNames = @(
    "AI_TW_Stock_Finance_Info", "AI_TW_Stock_Morning_Report", "AI_TW_Stock_AfterClose_Research",
    "AI_TW_Stock_Earnings_Daily", "AI_TW_Stock_CNYES_Nightly_News", "AI_TW_Stock_AfterClose_Research_Recovery"
)
foreach ($n in $OldNames) {
    $old = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
    if ($null -ne $old) { Unregister-ScheduledTask -TaskName $n -Confirm:$false; Write-Host "Removed old task: $n" }
}

$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Finance task: also run a missed slot as soon as the PC is available again (after sleep / power-off).
# Reports (morning / after-close) are NOT caught up on purpose - a morning mail at 20:00 is useless.
$FinanceSettings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited

Write-Host ""
Write-Host "============================================================"
Write-Host "AI TW Stock Research Scheduler v13"
Write-Host "============================================================"
Write-Host "Project : $ProjectDir"
Write-Host "Python  : $PythonPath"
Write-Host "User    : $CurrentUser"
Write-Host ""

foreach ($task in $Tasks) {
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "`"$Runner`" $($task.Job)" `
        -WorkingDirectory $ProjectDir

    $triggers = @()
    foreach ($t in $task.Times) {
        $parts = $t.Split(":")
        $at = [datetime]::Today.AddHours([double]$parts[0]).AddMinutes([double]$parts[1])
        $triggers += New-ScheduledTaskTrigger -Daily -At $at
    }

    $taskSettings = if (@("finance", "earnings") -contains $task.Job) { $FinanceSettings } else { $Settings }
    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $triggers `
        -Settings $taskSettings `
        -Principal $Principal `
        -Description "AI Taiwan stock scheduled job ($($task.Job))" `
        -Force | Out-Null

    Write-Host ("Registered : {0}" -f $task.Name)
    Write-Host ("Times      : {0}" -f ($task.Times -join ", "))
    Write-Host ""
}

# Catch-up task: 2 minutes after every logon (= after every reboot), refresh finance info
# only if the cache is older than 90 minutes.
$catchupAction = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "`"$Runner`" finance --if-stale-minutes 90" `
    -WorkingDirectory $ProjectDir
$logon = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
$logon.Delay = "PT2M"
Register-ScheduledTask `
    -TaskName "AI_TW_Stock_Finance_Catchup" `
    -Action $catchupAction `
    -Trigger $logon `
    -Settings $FinanceSettings `
    -Principal $Principal `
    -Description "Refresh finance info after logon/reboot if it is stale" `
    -Force | Out-Null
Write-Host "Registered : AI_TW_Stock_Finance_Catchup (2 min after every logon, only if cache older than 90 min)"
Write-Host ""

Write-Host "============================================================"
Write-Host "Done. Tasks are stored by Windows and survive shutdown / reboot."
Write-Host "  08:10 11:00 13:30 16:00 23:00       -> Finance info (news only)"
Write-Host "  18:00                               -> Earnings (news + earnings, once a day)"
Write-Host "  08:30                               -> Morning report"
Write-Host "  14:30                               -> After-close report"
Write-Host "============================================================"
Write-Host "Test now:  python .\scheduled_job_runner.py finance"
exit 0
