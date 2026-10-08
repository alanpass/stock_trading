# AI TW Stock Research Scheduler v11
# 08:30 Morning Report（先更新 finance_info_latest.json，再產生/寄送晨報）
# 14:30 After-close Research + Email
# 18:00 Finance info refresh
# 23:00 CNYES nightly crawl
# WakeToRun ON / no AtLogOn trigger / 不使用 StartWhenAvailable 追趕執行

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Runner = Join-Path $ProjectDir "scheduled_job_runner.py"

# 優先鎖定使用者專案目前使用的 Python 3.13；
# 不讓 Windows Store alias／其他 Conda Python 被排程誤選。
$PythonPath = $null
$PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -ne $PyLauncher) {
    try {
        $candidate = (& $PyLauncher.Source -3.13 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
        if ($candidate) {
            $candidate = $candidate.Trim()
            if (Test-Path -LiteralPath $candidate) { $PythonPath = $candidate }
        }
    } catch {}
}
if (-not $PythonPath) {
    $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) { $PythonCommand = Get-Command python -ErrorAction SilentlyContinue }
    if ($null -ne $PythonCommand -and (Test-Path -LiteralPath $PythonCommand.Source)) {
        $PythonPath = $PythonCommand.Source
    }
}
if (-not $PythonPath) {
    Write-Host "ERROR: Python 3.13 / python.exe not found." -ForegroundColor Red
    exit 1
}
$CurrentUser = if ([string]::IsNullOrWhiteSpace($env:USERDOMAIN)) { $env:USERNAME } else { "$env:USERDOMAIN\$env:USERNAME" }

if (-not (Test-Path -LiteralPath $Runner)) {
    Write-Host "ERROR: scheduled_job_runner.py not found: $Runner" -ForegroundColor Red
    exit 1
}

try {
    $version = & $PythonPath -c "import sys; print(sys.version)" 2>$null
    Write-Host "PythonVersion: $version"
} catch {}

$Tasks = @(
    @{ Name="AI_TW_Stock_Morning_Report"; Job="morning"; Hour=8;  Minute=30 },
    @{ Name="AI_TW_Stock_AfterClose_Research"; Job="afterclose"; Hour=14; Minute=30 },
    @{ Name="AI_TW_Stock_Finance_Info"; Job="finance"; Hour=18; Minute=0 },
    @{ Name="AI_TW_Stock_CNYES_Nightly_News"; Job="cnyes"; Hour=23; Minute=0 }
)

foreach ($task in $Tasks) {
    Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
}
Get-ScheduledTask -TaskName "AI_TW_Stock_AfterClose_Research_Recovery" -ErrorAction SilentlyContinue |
    Unregister-ScheduledTask -Confirm:$false

$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

Write-Host "============================================================"
Write-Host "AI TW Stock Research Scheduler v11"
Write-Host "============================================================"
Write-Host "Project : $ProjectDir"
Write-Host "Runner  : $Runner"
Write-Host "Python  : $PythonPath"
Write-Host "User    : $CurrentUser"
Write-Host ""

foreach ($task in $Tasks) {
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "`"$Runner`" $($task.Job)" `
        -WorkingDirectory $ProjectDir

    $triggerAt = [datetime]::Today.AddHours([double]$task.Hour).AddMinutes([double]$task.Minute)
    $trigger = New-ScheduledTaskTrigger -Daily -At $triggerAt

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "AI Taiwan stock scheduled job v11 ($($task.Job))" `
        -Force | Out-Null

    $created = Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue
    if ($null -eq $created) {
        Write-Host "FAILED: $($task.Name)" -ForegroundColor Red
        exit 2
    }
    $ti = Get-ScheduledTaskInfo -TaskName $task.Name
    Write-Host ("REGISTERED: {0} | State={1} | Next={2}" -f $task.Name, $created.State, $ti.NextRunTime)
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Scheduler setup completed"
Write-Host "============================================================"
Write-Host "08:30 -> Morning Report + Finance info update"
Write-Host "14:30 -> After-close Research + Email"
Write-Host "18:00 -> Finance info refresh"
Write-Host "23:00 -> CNYES nightly crawl"
Write-Host "WakeToRun         : ON"
Write-Host "StartWhenAvailable : OFF（未設定追趕啟動）"
Write-Host "AtLogOn           : NONE"
Write-Host ""
Write-Host "完全關機時不會開機後立即補跑錯過的任務；只等待下一次每日排程。"
exit 0
