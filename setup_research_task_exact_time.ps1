# ============================================================
# setup_research_task_exact_time.ps1
#
# AI TW Stock After-Close Research
#
# IMPORTANT:
# This version runs ONLY at the configured daily time.
# It does NOT create an AtLogOn recovery task.
# It also does NOT use StartWhenAvailable.
#
# Therefore:
#   - Windows login will NOT start the research.
#   - The research starts only at the scheduled time.
#   - Sleep/Hibernate may wake the PC because WakeToRun is enabled.
#   - A completely powered-off PC cannot run Python at the scheduled time.
# ============================================================

$ErrorActionPreference = "Stop"

# ============================================================
# EASY TIME SETTING
# ============================================================
# Change ONLY these two values.
#
# 15:00 -> $ScheduleHour = 15 ; $ScheduleMinute = 0
# 16:15 -> $ScheduleHour = 16 ; $ScheduleMinute = 15
# 21:30 -> $ScheduleHour = 21 ; $ScheduleMinute = 30
#
$ScheduleHour = 14
$ScheduleMinute = 0

# ============================================================
# Basic paths
# ============================================================

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResearchScript = Join-Path $ProjectDir "run_after_close_research.py"

$TaskName = "AI_TW_Stock_AfterClose_Research"
$OldRecoveryTaskName = "AI_TW_Stock_AfterClose_Research_Recovery"

Write-Host ""
Write-Host "=============================================="
Write-Host "AI TW Stock After-Close Research Setup"
Write-Host "=============================================="
Write-Host ""
Write-Host "Project : $ProjectDir"
Write-Host "Script  : $ResearchScript"
Write-Host ""

# ============================================================
# Check Python
# ============================================================

$PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue

if ($null -eq $PythonCommand) {
    Write-Host "ERROR: Python was not found in PATH." -ForegroundColor Red
    exit 1
}

$Python = $PythonCommand.Source

Write-Host "Python  : $Python"
Write-Host ""

# ============================================================
# Check research script
# ============================================================

if (-not (Test-Path -LiteralPath $ResearchScript)) {
    Write-Host "ERROR: run_after_close_research.py was not found." -ForegroundColor Red
    Write-Host $ResearchScript
    exit 1
}

# ============================================================
# Current user
# ============================================================

$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

Write-Host "User    : $CurrentUser"
Write-Host ""

# ============================================================
# Validate time
# ============================================================

if ($ScheduleHour -lt 0 -or $ScheduleHour -gt 23) {
    Write-Host "ERROR: ScheduleHour must be between 0 and 23." -ForegroundColor Red
    exit 1
}

if ($ScheduleMinute -lt 0 -or $ScheduleMinute -gt 59) {
    Write-Host "ERROR: ScheduleMinute must be between 0 and 59." -ForegroundColor Red
    exit 1
}

$ScheduleTime = [datetime]::Today.AddHours($ScheduleHour).AddMinutes($ScheduleMinute)

Write-Host "Daily schedule : $($ScheduleTime.ToString('HH:mm'))"
Write-Host ""

# ============================================================
# Action
# ============================================================

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$ResearchScript`"" `
    -WorkingDirectory $ProjectDir

# ============================================================
# Settings
#
# IMPORTANT:
# Do NOT use -StartWhenAvailable.
# That setting can run a missed task when Windows becomes
# available later.
#
# WakeToRun stays enabled for Sleep/Hibernate.
# ============================================================

$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# ============================================================
# Principal
# ============================================================

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

# ============================================================
# Daily trigger ONLY
# ============================================================

$DailyTrigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $ScheduleTime

# ============================================================
# Remove old daily task
# ============================================================

$OldTask = Get-ScheduledTask `
    -TaskName $TaskName `
    -ErrorAction SilentlyContinue

if ($null -ne $OldTask) {
    Write-Host "Removing old daily task..."
    Unregister-ScheduledTask `
        -TaskName $TaskName `
        -Confirm:$false
}

# ============================================================
# Remove old AtLogOn recovery task
#
# THIS IS THE IMPORTANT FIX.
# ============================================================

$OldRecoveryTask = Get-ScheduledTask `
    -TaskName $OldRecoveryTaskName `
    -ErrorAction SilentlyContinue

if ($null -ne $OldRecoveryTask) {
    Write-Host "Removing old login recovery task..."
    Unregister-ScheduledTask `
        -TaskName $OldRecoveryTaskName `
        -Confirm:$false
}

# ============================================================
# Register exact-time daily task
# ============================================================

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $DailyTrigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "AI Taiwan stock after-close research and email report at the configured daily time." `
    -Force | Out-Null

# ============================================================
# Verify
# ============================================================

$Task = Get-ScheduledTask `
    -TaskName $TaskName

$Info = Get-ScheduledTaskInfo `
    -TaskName $TaskName

$RecoveryStillExists = Get-ScheduledTask `
    -TaskName $OldRecoveryTaskName `
    -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=============================================="
Write-Host "Scheduled task created successfully"
Write-Host "=============================================="
Write-Host ""
Write-Host "Task name : $($Task.TaskName)"
Write-Host "State     : $($Task.State)"
Write-Host "Schedule  : Every day at $($ScheduleTime.ToString('HH:mm'))"
Write-Host "Next run  : $($Info.NextRunTime)"
Write-Host "Last run  : $($Info.LastRunTime)"
Write-Host "Result    : $($Info.LastTaskResult)"
Write-Host ""

if ($null -eq $RecoveryStillExists) {
    Write-Host "Login recovery task : REMOVED"
} else {
    Write-Host "WARNING: Login recovery task still exists." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Behavior"
Write-Host "--------"
Write-Host "Windows login       : NO research execution"
Write-Host "Daily schedule      : $($ScheduleTime.ToString('HH:mm'))"
Write-Host "Sleep/Hibernate     : WakeToRun enabled"
Write-Host "Missed schedule     : No automatic catch-up"
Write-Host "Full shutdown       : Cannot execute while powered off"
Write-Host ""
Write-Host "Setup completed."
Write-Host ""
