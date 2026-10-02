# AI TW Stock Research Assistant - Windows Scheduler Setup v9.6
# 08:30 Morning / 14:30 AfterClose / 23:00 CNYES
# WakeToRun ON, StartWhenAvailable OFF, AtLogOn NONE.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Runner = Join-Path $ProjectDir "scheduled_job_runner.py"

# Prefer the real Python 3.13 interpreter.
$PythonPath = $null
$pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -ne $pyLauncher) {
    try {
        $candidate = (& $pyLauncher.Source -3.13 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            $PythonPath = $candidate.Trim()
        }
    } catch {}
}
if (-not $PythonPath) {
    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd -and (Test-Path -LiteralPath $pythonCmd.Source)) {
        $PythonPath = $pythonCmd.Source
    }
}
if (-not $PythonPath) {
    Write-Host "ERROR: Python 3.13 / python.exe not found." -ForegroundColor Red
    exit 1
}

$PythonDir = Split-Path -Parent $PythonPath
$PythonwPath = Join-Path $PythonDir "pythonw.exe"
$TaskPythonPath = if (Test-Path -LiteralPath $PythonwPath) { $PythonwPath } else { $PythonPath }

if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py not found: $Runner" -ForegroundColor Red
    exit 1
}
foreach ($job in @("run_morning_report.py","run_after_close_research.py","run_cnyes_news_nightly.py")) {
    $path = Join-Path $ProjectDir $job
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "ERROR: script not found: $path" -ForegroundColor Red
        exit 1
    }
}

$CurrentUser = if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) { $env:USERNAME } else { "$env:USERDOMAIN\$env:USERNAME" }
$Tasks = @(
    @{ Name="AI_TW_Stock_CNYES_Nightly_News"; Job="cnyes"; Hour=23; Minute=0 },
    @{ Name="AI_TW_Stock_Morning_Report"; Job="morning"; Hour=8; Minute=30 },
    @{ Name="AI_TW_Stock_AfterClose_Research"; Job="afterclose"; Hour=14; Minute=30 }
)

Write-Host "============================================================"
Write-Host "AI TW Stock Scheduler Setup v9.6"
Write-Host "============================================================"
Write-Host "Project : $ProjectDir"
Write-Host "Python  : $PythonPath"
Write-Host "TaskPy  : $TaskPythonPath"
Write-Host "Runner  : $Runner"
Write-Host "User    : $CurrentUser"
Write-Host ""

foreach ($task in $Tasks) {
    Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
}
Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false

$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

foreach ($task in $Tasks) {
    $triggerAt = [datetime]::Today.AddHours([double]$task.Hour).AddMinutes([double]$task.Minute)
    $trigger = New-ScheduledTaskTrigger -Daily -At $triggerAt
    # pythonw prevents the black console window from flashing.
    $argument = '-u "{0}" {1}' -f $Runner, $task.Job
    $action = New-ScheduledTaskAction -Execute $TaskPythonPath -Argument $argument -WorkingDirectory $ProjectDir

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI Taiwan Stock Research scheduled job v9.6 ($($task.Job))" `
        -Force | Out-Null
}

Write-Host ""
Write-Host "Verifying..."
$ok=$true
foreach ($task in $Tasks) {
    try {
        $t=Get-ScheduledTask -TaskName $task.Name -ErrorAction Stop
        $i=Get-ScheduledTaskInfo -TaskName $task.Name -ErrorAction Stop
        Write-Host "$($task.Name) : OK" -ForegroundColor Green
        Write-Host "  State       : $($t.State)"
        Write-Host "  Action      : $($t.Actions.Execute)"
        Write-Host "  Arguments   : $($t.Actions.Arguments)"
        Write-Host "  WorkingDir  : $($t.Actions.WorkingDirectory)"
        Write-Host "  NextRunTime : $($i.NextRunTime)"
        Write-Host "  LastResult  : $($i.LastTaskResult)"
        Write-Host ""
    } catch {
        Write-Host "$($task.Name) : FAILED - $($_.Exception.Message)" -ForegroundColor Red
        $ok=$false
    }
}

$recovery=Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue
if ($null -ne $recovery) {
    Write-Host "ERROR: legacy recovery task still exists." -ForegroundColor Red
    $ok=$false
}

if (-not $ok) { exit 1 }
Write-Host "============================================================"
Write-Host "Scheduler setup completed."
Write-Host "08:30 Morning | 14:30 AfterClose | 23:00 CNYES"
Write-Host "WakeToRun=ON | StartWhenAvailable=OFF | AtLogOn=NONE"
Write-Host "============================================================"
exit 0
