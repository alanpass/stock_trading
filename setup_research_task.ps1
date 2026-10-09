# Windows Task Scheduler setup for AI TW Stock Research Assistant
# v10.0 - six daily finance refresh/publish jobs
# This file intentionally uses ASCII-only text for Windows PowerShell compatibility.

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
}
if ($null -eq $PythonCommand) {
    Write-Host "ERROR: python.exe was not found in PATH." -ForegroundColor Red
    exit 1
}

$PythonPath = $PythonCommand.Source

$Tasks = @(
    @{
        Name = "AI_TW_Stock_Finance_Update_0810"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 8
        Minute = 10
    },
    @{
        Name = "AI_TW_Stock_Finance_Update_1100"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 11
        Minute = 0
    },
    @{
        Name = "AI_TW_Stock_Finance_Update_1330"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 13
        Minute = 30
    },
    @{
        Name = "AI_TW_Stock_Finance_Update_1600"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 16
        Minute = 0
    },
    @{
        Name = "AI_TW_Stock_Finance_Update_1800"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 18
        Minute = 0
    },
    @{
        Name = "AI_TW_Stock_Finance_Update_2300"
        Script = Join-Path $ProjectDir "scheduled_job_runner.py"
        Arguments = "finance"
        Hour = 23
        Minute = 0
    },
    @{
        Name = "AI_TW_Stock_Morning_Report"
        Script = Join-Path $ProjectDir "run_morning_report.py"
        Hour = 8
        Minute = 30
    },
    @{
        Name = "AI_TW_Stock_AfterClose_Research"
        Script = Join-Path $ProjectDir "run_after_close_research.py"
        Hour = 14
        Minute = 30
    }
)

Write-Host ""
Write-Host "============================================================"
Write-Host "AI TW Stock Research Assistant - Scheduler Setup v10.0"
Write-Host "============================================================"
Write-Host ""
Write-Host "Project : $ProjectDir"
Write-Host "Python  : $PythonPath"
Write-Host ""

# Current user, used with Interactive logon.
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"
if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) {
    $CurrentUser = $env:USERNAME
}
Write-Host "User    : $CurrentUser"
Write-Host ""

# Remove the legacy standalone crawler task; the 23:00 finance job now runs
# the full crawler + AI summary + GitHub publication pipeline.
$LegacyTaskName = "AI_TW_Stock_CNYES_Nightly_News"
$LegacyTask = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
if ($null -ne $LegacyTask) {
    Unregister-ScheduledTask -TaskName $LegacyTaskName -Confirm:$false
    Write-Host "Removed legacy task: $LegacyTaskName"
}

foreach ($task in $Tasks) {
    if (-not (Test-Path -LiteralPath $task.Script)) {
        Write-Host "ERROR: Script not found: $($task.Script)" -ForegroundColor Red
        exit 1
    }
}

# Important:
# - WakeToRun = enabled
# - StartWhenAvailable = NOT enabled
# - No AtLogOn recovery trigger
# Therefore a missed run is NOT started immediately after login.
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

foreach ($task in $Tasks) {
    Write-Host "Creating: $($task.Name) at $($task.Hour.ToString('00')):$($task.Minute.ToString('00'))"

    $Time = [datetime]::Today.AddHours($task.Hour).AddMinutes($task.Minute)
    $Trigger = New-ScheduledTaskTrigger -Daily -At $Time

    $ScriptArguments = ""
    if ($task.ContainsKey("Arguments")) {
        $ScriptArguments = [string]$task.Arguments
    }

    $Argument = '"{0}"' -f $task.Script
    if (-not [string]::IsNullOrWhiteSpace($ScriptArguments)) {
        $Argument += " $ScriptArguments"
    }

    $Action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument $Argument `
        -WorkingDirectory $ProjectDir

    $TaskObject = New-ScheduledTask `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI TW Stock Research Assistant: $($task.Name)"

    Register-ScheduledTask `
        -TaskName $task.Name `
        -InputObject $TaskObject `
        -Force | Out-Null
}

Write-Host ""
Write-Host "Verifying scheduled tasks..."
Write-Host ""

$allOk = $true

foreach ($task in $Tasks) {
    try {
        $info = Get-ScheduledTask -TaskName $task.Name -ErrorAction Stop
        $taskInfo = Get-ScheduledTaskInfo -TaskName $task.Name -ErrorAction Stop

        Write-Host ("{0} : OK | State={1} | NextRun={2}" -f `
            $task.Name, `
            $info.State, `
            $taskInfo.NextRunTime)

        if (-not $info) {
            $allOk = $false
        }
    }
    catch {
        Write-Host ("{0} : FAILED | {1}" -f $task.Name, $_.Exception.Message) -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host ""

if (-not $allOk) {
    Write-Host "Scheduler setup FAILED. The task list above contains the error." -ForegroundColor Red
    exit 1
}

Write-Host "Scheduler setup completed successfully." -ForegroundColor Green
Write-Host ""
Write-Host "08:10, 11:00, 13:30, 16:00, 18:00, 23:00  Finance refresh + GitHub publish"
Write-Host "08:30  Morning report"
Write-Host "14:30  After-close report"
Write-Host ""
Write-Host "StartWhenAvailable = OFF"
Write-Host "WakeToRun          = ON"
Write-Host "No AtLogOn trigger is created."
Write-Host ""
exit 0
