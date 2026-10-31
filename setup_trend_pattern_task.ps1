#requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = "StockTrading-TrendPatterns-1900"
$Updater = Join-Path $ProjectRoot "run_trend_pattern_update.py"

if (!(Test-Path -LiteralPath $Updater)) {
    throw "找不到 $Updater。請先同步最新版專案檔案。"
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $PythonExe = $VenvPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (!$PythonCommand) {
        throw "找不到 Python。請先安裝 Python 並加入 PATH，或在專案建立 .venv。"
    }
    $PythonExe = $PythonCommand.Source
}

$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument ('"' + $Updater + '"') -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At ([datetime]::Today.AddHours(19))
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "台灣時間每週一至週五 19:00 檢查全部產業概念股趨勢型態；非交易日跳過，交易日每天發布。" -Force | Out-Null

Write-Host ""
Write-Host "已建立排程：$TaskName" -ForegroundColor Green
Write-Host "執行時間：每週一至週五 19:00（程式會再以官方交易日曆排除休市日）"
Write-Host "Python：$PythonExe"
Write-Host "程式：$Updater"
Write-Host "Log：$(Join-Path $ProjectRoot 'output\scheduler_logs\trend_patterns_runner.log')"
Write-Host ""
Write-Host "請確認使用者環境變數 GITHUB_TOKEN 已設定，且具有 alanpass/stock_trading 的 Contents: Read and write 權限。"
Write-Host "注意：Windows 需開機且該使用者工作階段可執行排程；若電腦關機，無法準時執行。"
