# ============================================================
# setup_github_auto_publish.ps1
#
# 檢查 Windows 本機財經研究專案是否已準備好自動發布：
#   Python / Git / repository / origin / branch / Git Credential Manager
#
# 這支腳本不會把 Token 寫進程式碼，也不會把 Token 存進 .env。
# 認證交給 Git Credential Manager 或 SSH。
# ============================================================
$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectDir

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "AI TW Stock GitHub Auto-Publish Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host ""

$Git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $Git) {
    Write-Host "ERROR: 找不到 git.exe。請先安裝 Git for Windows。" -ForegroundColor Red
    exit 1
}
Write-Host "Git: $($Git.Source)"

$Python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    Write-Host "ERROR: 找不到 python.exe。" -ForegroundColor Red
    exit 1
}
Write-Host "Python: $($Python.Source)"

try {
    git rev-parse --show-toplevel | Out-Null
} catch {
    Write-Host "ERROR: 目前資料夾不是 Git repository。" -ForegroundColor Red
    Write-Host "請先把這個專案 clone 到本機，或在專案目錄執行 git init。"
    exit 1
}

$Root = (git rev-parse --show-toplevel).Trim()
Write-Host "Repository root: $Root"

$Origin = ""
try { $Origin = (git remote get-url origin 2>$null).Trim() } catch {}
if ([string]::IsNullOrWhiteSpace($Origin)) {
    Write-Host "WARNING: 尚未設定 origin。" -ForegroundColor Yellow
    Write-Host "請設定：git remote add origin https://github.com/你的帳號/你的Repository.git"
} else {
    # 不完整顯示可能含有機密資訊的 remote URL。
    Write-Host "Origin: 已設定（URL 已隱藏）" -ForegroundColor Green
}

$Branch = ""
try { $Branch = (git branch --show-current).Trim() } catch {}
if ([string]::IsNullOrWhiteSpace($Branch)) {
    Write-Host "ERROR: 目前不是正常分支（可能是 detached HEAD）。" -ForegroundColor Red
    exit 1
}
Write-Host "Branch: $Branch"

$Helper = ""
try { $Helper = (git config --global credential.helper).Trim() } catch {}
if ([string]::IsNullOrWhiteSpace($Helper)) {
    Write-Host "WARNING: 尚未偵測到 Git credential helper。" -ForegroundColor Yellow
    Write-Host "建議執行：git config --global credential.helper manager"
} else {
    Write-Host "Credential helper: $Helper" -ForegroundColor Green
}

$Publisher = Join-Path $ProjectDir "publish_research_data.py"
if (-not (Test-Path -LiteralPath $Publisher)) {
    Write-Host "ERROR: publish_research_data.py 不存在。" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "執行公開資料發布器 dry-run..." -ForegroundColor Cyan
& $Python.Source -u $Publisher --dry-run
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: dry-run 失敗，請先修正上面的錯誤。" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "GitHub auto-publish 基礎檢查完成" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "接下來請確認 GitHub 登入已完成，之後由 scheduled_job_runner.py 自動發布。"
Write-Host "若要暫停自動發布，可設定環境變數：`$env:AUTO_GIT_PUBLISH='false'"
Write-Host ""
