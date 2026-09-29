$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = 'C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard'
if (-not (Test-Path $target)) { throw "找不到 stock_dashboard：$target" }
$backup = Join-Path $target ("backup_v72_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $backup | Out-Null
foreach ($f in @('earnings_call_agent.py','stock_dashboard.py','test_earnings_agent.py')) {
    if (Test-Path (Join-Path $target $f)) { Copy-Item (Join-Path $target $f) (Join-Path $backup $f) -Force }
    Copy-Item (Join-Path $root $f) (Join-Path $target $f) -Force
}
Write-Host "V72 已套用。原檔備份：$backup" -ForegroundColor Green
Write-Host "未修改 output/data/models/.env" -ForegroundColor Green
