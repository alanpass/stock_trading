# ============================================================
# setup_research_task_v10.ps1
# AI TW Stock Research System
#
# Schedule:
#   08:30  Morning report + finance cache refresh
#   14:30  After-close research + email
#   18:00  Finance info refresh
#   23:00  CNYES nightly news crawl
#
# Important:
#   - NO AtLogOn recovery task
#   - NO StartWhenAvailable
#   - WakeToRun = ON
#   - Sleep/Hibernate may wake the PC
#   - Completely powered-off PC cannot execute Python at the exact time
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Runner = Join-Path $ProjectDir "scheduled_job_runner.py"

$Tasks = @(
    @{ Name="AI_TW_Stock_Morning_Report";  Job="morning";    Hour=8;  Minute=30 },
    @{ Name="AI_TW_Stock_AfterClose_Research"; Job="afterclose"; Hour=14; Minute=30 },
    @{ Name="AI_TW_Stock_Finance_Info"; Job="finance"; Hour=18; Minute=0 },
    @{ Name="AI_TW_Stock_CNYES_Nightly_News"; Job="cnyes"; Hour=23; Minute=0 }
)

Write-Host ""
Write-Host "============================================================"
Write-Host "AI TW Stock Research Scheduler v10"
Write-Host "============================================================"
Write-Host "Project : $ProjectDir"
Write-Host "Runner  : $Runner"
Write-Host ""

if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py not found." -ForegroundColor Red
    exit 2
}

# ------------------------------------------------------------
# Resolve Python 3.13
# ------------------------------------------------------------
$Python = $null

try {
    $py = Get-Command py.exe -ErrorAction Stop
    $candidate = (& $py.Source -3.13 -c "import sys; print(sys.executable)").Trim()
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        $Python = $candidate
    }
} catch {}

if (-not $Python) {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        $Python = $cmd.Source
    }
}

if (-not $Python) {
    Write-Host "ERROR: Python was not found." -ForegroundColor Red
    exit 2
}

Write-Host "Python  : $Python"
Write-Host ""

# ------------------------------------------------------------
# Current interactive Windows user
# ------------------------------------------------------------
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "User    : $CurrentUser"
Write-Host ""

# ------------------------------------------------------------
# Settings: exact-time + wake
# ------------------------------------------------------------
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

foreach ($item in $Tasks) {
    $TaskName = $item.Name
    $Job = $item.Job
    $Boundary = [datetime]::Today.AddHours($item.Hour).AddMinutes($item.Minute)

    Write-Host "Register : $TaskName"
    Write-Host "Time     : $($Boundary.ToString('HH:mm'))"
    Write-Host "Job      : $Job"

    $Action = New-ScheduledTaskAction `
        -Execute $Python `
        -Argument "`"$Runner`" $Job" `
        -WorkingDirectory $ProjectDir

    $Trigger = New-ScheduledTaskTrigger -Daily -At $Boundary

    $old = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $old) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI TW Stock Research - $Job" | Out-Null

    Write-Host "OK"
    Write-Host ""
}

# Remove legacy recovery task(s)
$LegacyNames = @(
    "AI_TW_Stock_AfterClose_Research_Recovery",
    "AI_TW_Stock_Morning_Report_Recovery",
    "AI_TW_Stock_Finance_Info_Recovery"
)

foreach ($name in $LegacyNames) {
    $old = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -ne $old) {
        Write-Host "Removing legacy task: $name"
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
}

Write-Host "============================================================"
Write-Host "Scheduler setup completed."
Write-Host "============================================================"
Write-Host ""
Write-Host "08:30 -> Morning + Finance"
Write-Host "14:30 -> After-close research + Email"
Write-Host "18:00 -> Finance info refresh"
Write-Host "23:00 -> CNYES nightly crawl"
Write-Host ""
Write-Host "WakeToRun       : ON"
Write-Host "StartWhenAvailable: OFF"
Write-Host "AtLogOn         : NONE"
Write-Host ""
