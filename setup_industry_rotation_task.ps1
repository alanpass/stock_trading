#requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = "StockTrading-IndustryAnalysis-1510"
$Updater = Join-Path $ProjectRoot "run_industry_rotation_update.py"

if (!(Test-Path -LiteralPath $Updater)) {
    throw "找不到 $Updater。請先同步最新版專案檔案。"
}

# 優先使用專案虛擬環境；否則使用 PATH 中的 Python。
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
# 每週一至週五 15:10 啟動；程式本身會用官方交易日曆排除國定休市日。
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At ([datetime]::Today.AddHours(15).AddMinutes(10))
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "台股交易日 15:10 更新產業分析輪動資料並發布到 GitHub。" -Force | Out-Null

Write-Host ""
Write-Host "已建立排程：$TaskName" -ForegroundColor Green
Write-Host "執行時間：每週一至週五 15:10（休市日由程式自動略過）"
Write-Host "Python：$PythonExe"
Write-Host "程式：$Updater"
Write-Host "Log：$(Join-Path $ProjectRoot 'logs\industry_rotation_task.log')"
Write-Host ""
Write-Host "注意：電腦需開機且 Windows 使用者需登入；若電腦關機，無法準時執行排程。"
Write-Host "請確認使用者環境變數 GITHUB_TOKEN 已設定，且具備 Repository Contents 寫入權限。"
