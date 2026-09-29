$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stockTrading = Split-Path -Parent $root
$project = Join-Path $stockTrading 'stock_dashboard'
$backup = Join-Path $stockTrading 'stock_dashboard_backup_20260925'
$code = Join-Path $root 'code'

if (-not (Test-Path $project)) { throw "找不到專案：$project" }

Write-Host '=== 1. 備份目前程式碼 ==='
$backupCode = Join-Path $stockTrading ('stock_dashboard_code_backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $backupCode | Out-Null
Get-ChildItem $project -File -Filter '*.py' | Copy-Item -Destination $backupCode -Force
Write-Host "目前 .py 已備份到 $backupCode"

Write-Host '=== 2. 套用安全版 v67 程式碼 ==='
Copy-Item (Join-Path $code '*.py') -Destination $project -Force
Remove-Item (Join-Path $project '__pycache__') -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path $backup) {
    Write-Host '=== 3. 從原 v65 備份恢復執行資料 ==='
    foreach ($name in @('output','models','data')) {
        $src = Join-Path $backup $name
        $dst = Join-Path $project $name
        if (Test-Path $src) {
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            Copy-Item (Join-Path $src '*') -Destination $dst -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "恢復 $name"
        }
    }
    foreach ($name in @('.env','user_watchlist.json')) {
        $src = Join-Path $backup $name
        $dst = Join-Path $project $name
        if (Test-Path $src) {
            Copy-Item $src -Destination $dst -Force
            Write-Host "恢復 $name"
        }
    }
} else {
    Write-Warning "找不到 $backup；沒有自動覆蓋目前 output/models/data/.env。"
    Write-Warning '代表你沒有建立前一步的 backup，請保留目前執行資料，先直接啟動測試。'
}

Write-Host '=== 4. Python 語法檢查 ==='
Push-Location $project
& python -m py_compile earnings_call_agent.py test_earnings_agent.py stock_dashboard.py research_agent.py
if ($LASTEXITCODE -ne 0) { throw 'Python 語法檢查失敗' }
Pop-Location

Write-Host ''
Write-Host '修復完成。接著：'
Write-Host 'cd C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard'
Write-Host 'python test_earnings_agent.py 6582 --crawl-only'
Write-Host 'python -m streamlit run stock_dashboard.py'
