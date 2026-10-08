# AI TW Stock Research Scheduler v10
# 08:30 Morning
# 14:30 After-close
# 18:00 Finance info refresh
# 23:00 CNYES nightly
# WakeToRun ON, StartWhenAvailable OFF, AtLogOn NONE.

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

$CurrentUser = if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) { $env:USERNAME } else { "$env:USERDOMAIN\$env:USERNAME" }

$Tasks = @(
    @{ Name="AI_TW_Stock_Morning_Report"; Job="morning"; Hour=8;  Minute=30 },
    @{ Name="AI_TW_Stock_AfterClose_Research"; Job="afterclose"; Hour=14; Minute=30 },
    @{ Name="AI_TW_Stock_Finance_Info"; Job="finance" },
    @{ Name="AI_TW_Stock_CNYES_Nightly_News"; Job="cnyes"; Hour=23; Minute=0 }
)

if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py not found: $Runner" -ForegroundColor Red
    exit 1
}

foreach ($task in $Tasks) {
    $old = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
    if ($null -ne $old) { Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false }
}
Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue |
    Unregister-ScheduledTask -Confirm:$false

$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

Write-Host ""
Write-Host "============================================================"
Write-Host "AI TW Stock Research Scheduler v10"
Write-Host "============================================================"
Write-Host "Project : $ProjectDir"
Write-Host "Runner  : $Runner"
Write-Host "Python  : $PythonPath"
Write-Host "User    : $CurrentUser"
Write-Host ""

foreach ($task in $Tasks) {
    $args = $task.Job
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "`"$Runner`" $args" `
        -WorkingDirectory $ProjectDir
    if ($task.Job -eq "finance") {
        $triggerAtMorning = [datetime]::Today.AddHours(8).AddMinutes(30)
        $triggerAtEvening = [datetime]::Today.AddHours(18)
        $triggers = @(
            (New-ScheduledTaskTrigger -Daily -At $triggerAtMorning),
            (New-ScheduledTaskTrigger -Daily -At $triggerAtEvening)
        )
    } else {
        $triggerAt = [datetime]::Today.AddHours([double]$task.Hour).AddMinutes([double]$task.Minute)
        $triggers = @(New-ScheduledTaskTrigger -Daily -At $triggerAt)
    }

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $triggers `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI Taiwan stock scheduled job ($($task.Job))" `
        -Force | Out-Null

    Write-Host ("Register : {0}" -f $task.Name)
    if ($task.Job -eq "finance") {
        Write-Host "Time     : 08:30, 18:00"
    } else {
        Write-Host ("Time     : {0:00}:{1:00}" -f $task.Hour, $task.Minute)
    }
    Write-Host ("Job      : {0}" -f $task.Job)
    Write-Host "OK"
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "Scheduler setup completed."
Write-Host "============================================================"
Write-Host "08:30 -> Morning + Finance info refresh"
Write-Host "14:30 -> After-close research + Email"
Write-Host "18:00 -> Finance info refresh"
Write-Host "23:00 -> CNYES nightly crawl"
Write-Host ""
Write-Host "WakeToRun        : ON"
Write-Host "StartWhenAvailable: OFF"
Write-Host "AtLogOn          : NONE"
Write-Host ""
exit 0
