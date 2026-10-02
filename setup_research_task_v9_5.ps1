# AI TW Stock Research Assistant - Windows Scheduler Setup v9.5
#
# Schedule:
#   08:30  Morning report
#   14:30  After-close research
#   23:00  CNYES nightly news
#
# Behavior:
#   WakeToRun          = ON
#   StartWhenAvailable = OFF
#   AtLogOn trigger    = NONE
#
# All tasks run through scheduled_job_runner.py so stdout/stderr and
# output validation are saved under output\scheduler_logs.

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
$Runner = Join-Path $ProjectDir "scheduled_job_runner.py"

$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"
if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) {
    $CurrentUser = $env:USERNAME
}

$Tasks = @(
    @{ Name = "AI_TW_Stock_CNYES_Nightly_News"; Job = "cnyes"; Hour = 23; Minute = 0 },
    @{ Name = "AI_TW_Stock_Morning_Report"; Job = "morning"; Hour = 8; Minute = 30 },
    @{ Name = "AI_TW_Stock_AfterClose_Research"; Job = "afterclose"; Hour = 14; Minute = 30 }
)

Write-Host ""
Write-Host "============================================================"
Write-Host "AI TW Stock Research Assistant - Scheduler Setup v9.5"
Write-Host "============================================================"
Write-Host ""
Write-Host "Project : $ProjectDir"
Write-Host "Python  : $PythonPath"
Write-Host "Runner  : $Runner"
Write-Host "User    : $CurrentUser"
Write-Host ""

if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py was not found." -ForegroundColor Red
    exit 1
}

foreach ($task in $Tasks) {
    $expected = Join-Path $ProjectDir $task.Job
    Write-Host ("Registering {0} at {1}:{2:00}" -f $task.Name, $task.Hour, $task.Minute)
}

# Remove old scheduled tasks, including the legacy login-recovery task.
foreach ($task in $Tasks) {
    Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
}
Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue |
    Unregister-ScheduledTask -Confirm:$false

# Keep the user's requested behavior:
# - wake sleeping/hibernating PC when possible
# - do not catch up immediately after login / startup
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

foreach ($task in $Tasks) {
    $triggerAt = [datetime]::Today.AddHours([double]$task.Hour).AddMinutes([double]$task.Minute)
    $trigger = New-ScheduledTaskTrigger -Daily -At $triggerAt
    $argument = '-u "{0}" {1}' -f $Runner, $task.Job
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument $argument `
        -WorkingDirectory $ProjectDir

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI Taiwan stock research scheduled job ($($task.Job))" `
        -Force | Out-Null
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Verifying scheduled tasks"
Write-Host "============================================================"
Write-Host ""

$allOk = $true
foreach ($task in $Tasks) {
    try {
        $registered = Get-ScheduledTask -TaskName $task.Name -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskName $task.Name -ErrorAction Stop
        $trigger = $registered.Triggers | Select-Object -First 1
        Write-Host ("{0} : OK" -f $task.Name) -ForegroundColor Green
        Write-Host ("  State       : {0}" -f $registered.State)
        Write-Host ("  LastRunTime : {0}" -f $info.LastRunTime)
        Write-Host ("  NextRunTime : {0}" -f $info.NextRunTime)
        Write-Host ("  LastResult  : {0}" -f $info.LastTaskResult)
        Write-Host ("  Action      : {0}" -f $registered.Actions.Execute)
        Write-Host ("  Arguments   : {0}" -f $registered.Actions.Arguments)
        Write-Host ("  Trigger     : {0}" -f $trigger.StartBoundary)
        Write-Host ""
    }
    catch {
        Write-Host ("{0} : FAILED | {1}" -f $task.Name, $_.Exception.Message) -ForegroundColor Red
        $allOk = $false
    }
}

$legacyRecovery = Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue
if ($null -ne $legacyRecovery) {
    Write-Host "ERROR: Legacy login-recovery task still exists." -ForegroundColor Red
    $allOk = $false
}

if (-not $allOk) {
    Write-Host "Scheduler setup FAILED." -ForegroundColor Red
    exit 1
}

Write-Host "============================================================"
Write-Host "Scheduler setup completed successfully."
Write-Host "============================================================"
Write-Host "08:30  Morning report"
Write-Host "14:30  After-close research"
Write-Host "23:00  CNYES nightly crawler"
Write-Host "WakeToRun          = ON"
Write-Host "StartWhenAvailable = OFF"
Write-Host "AtLogOn            = NONE"
Write-Host "Logs               = output\scheduler_logs"
Write-Host ""
exit 0
