$ErrorActionPreference = 'Stop'
$patchRoot = $PSScriptRoot
$projectRoot = Split-Path $patchRoot -Parent
$target = Join-Path $projectRoot 'stock_dashboard'
$src = Join-Path $patchRoot 'stock_dashboard'

if (-not (Test-Path $target)) {
    throw "找不到目前專案：$target。請確認 repair_v71 資料夾位於 stock_trading 根目錄內。"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $projectRoot ("stock_dashboard_code_backup_v71_" + $stamp)
New-Item -ItemType Directory -Path $backup | Out-Null

foreach ($name in @('earnings_call_agent.py','stock_dashboard.py','test_earnings_agent.py')) {
    $srcFile = Join-Path $src $name
    $dstFile = Join-Path $target $name
    if (-not (Test-Path $srcFile)) { throw "Patch 缺少檔案：$srcFile" }
    if (Test-Path $dstFile) { Copy-Item $dstFile (Join-Path $backup $name) -Force }
    Copy-Item $srcFile $dstFile -Force
}

Write-Host "v71 已安全套用。"
Write-Host "程式碼備份：$backup"
Write-Host "output / data / models / .env 沒有被修改。"
