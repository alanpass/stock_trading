#requires -Version 5.1
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = 'alanpass/stock_trading'
$Branch = 'main'
$RawBase = "https://raw.githubusercontent.com/$Repo/$Branch"
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$BackupDir = Join-Path $ProjectRoot "backup_before_industry_analysis_$Timestamp"
$StageDir = Join-Path $ProjectRoot ".industry_analysis_stage_$Timestamp"
$TaskSetupName = 'setup_industry_rotation_task.ps1'

$Files = @(
    @{ Name = 'stock_dashboard.py'; Markers = @('from industry_rotation import', 'from us_industry_indicators import', 'def _render_published_rotation(payload: dict)', 'rotation_published_avg_return_', '族群平均漲跌幅', '04｜美股產業指標股資訊', 'rotation_us_industry_category_v1', 'Styler.map', 'styled_frame', '查看全部 27 個產業主題與成分股分類', 'rotation_published_groups_all_v3', 'rrg_plot_groups', 'rotation_published_period_v3', 'quadrant_options = ["領先", "轉弱", "改善", "落後", "全選"]', 'width=1400, height=760, autosize=False', 'modeBarButtonsToRemove', 'rgba(252, 232, 232, 0.58)', '"doubleClick": "reset"', '"scrollZoom": False', 'hovermode="x unified"', '雙擊會重設回原始完整圖', '**台股｜產業族群平均漲跌幅**', '**美股｜對應指標股平均漲跌幅**', '產業分析') },
    @{ Name = 'industry_rotation.py'; Markers = @('ROTATION_THEMES', 'ROTATION_DISPLAY_NAMES', 'today_return', 'yesterday_return', 'def update_industry_rotation', 'TAIEX_MONTHLY_URL') },
    @{ Name = 'us_industry_indicators.py'; Markers = @('US_INDUSTRY_INDICATORS', 'get_us_indicator_symbols', 'fetch_us_indicators', 'Yahoo Finance chart', 'TSM') },
    @{ Name = 'run_industry_rotation_update.py'; Markers = @('def main()', 'update_industry_rotation') },
    @{ Name = 'setup_industry_rotation_task.ps1'; Markers = @('New-ScheduledTaskTrigger', '15:10') },
    @{ Name = 'INDUSTRY_ANALYSIS_SETUP.md'; Markers = @('產業族群平均漲跌幅', '每日更新架構', 'GITHUB_TOKEN') }
)

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host '台股產業分析 v13：本地端安全更新' -ForegroundColor Cyan
    Write-Host "專案位置：$ProjectRoot"
    Write-Host "備份位置：$BackupDir"
    Write-Host '============================================================' -ForegroundColor Cyan

    foreach ($File in $Files) {
        $Url = "$RawBase/$($File.Name)"
        $StagePath = Join-Path $StageDir $File.Name
        Write-Host "下載 $($File.Name) ..."
        Invoke-WebRequest -Uri $Url -OutFile $StagePath -UseBasicParsing -Headers @{ 'User-Agent' = 'Mozilla/5.0 StockTradingLocalUpdater/1.0'; 'Cache-Control' = 'no-cache' }
        if (!(Test-Path -LiteralPath $StagePath) -or (Get-Item -LiteralPath $StagePath).Length -lt 50) {
            throw "下載失敗或檔案過小：$($File.Name)"
        }
        $Text = Get-Content -LiteralPath $StagePath -Raw -Encoding UTF8
        foreach ($Marker in $File.Markers) {
            if (!$Text.Contains($Marker)) {
                throw "版本驗證失敗：$($File.Name) 缺少必要內容 [$Marker]。本地檔案尚未覆蓋。"
            }
        }
    }

    $PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $PythonExe)) {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (!$PythonCommand) { throw '找不到 Python；請先啟用專案使用的 Python 環境。' }
        $PythonExe = $PythonCommand.Source
    }

    # PowerShell 5.1 可能用系統 ANSI 編碼解析沒有 BOM 的 UTF-8 腳本。
    # 將下載的排程腳本明確重寫成 UTF-8 BOM，再用 PowerShell Parser 預先檢查。
    $SetupStagePath = Join-Path $StageDir $TaskSetupName
    $SetupText = Get-Content -LiteralPath $SetupStagePath -Raw -Encoding UTF8
    $Utf8Bom = New-Object System.Text.UTF8Encoding -ArgumentList $true
    [System.IO.File]::WriteAllText($SetupStagePath, $SetupText, $Utf8Bom)
    $PsTokens = $null
    $PsParseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($SetupStagePath, [ref]$PsTokens, [ref]$PsParseErrors) | Out-Null
    if ($PsParseErrors -and $PsParseErrors.Count -gt 0) {
        $PsErrorText = ($PsParseErrors | ForEach-Object { $_.Message }) -join '; '
        throw "PowerShell 語法檢查失敗：$PsErrorText；本地正式檔案尚未覆蓋。"
    }
    Write-Host 'PowerShell 語法檢查通過：setup_industry_rotation_task.ps1' -ForegroundColor Green

    # 在覆蓋任何正式檔案前，先確認所有 Python 檔案可解析。
    # 不使用 python -c 內嵌程式碼，避免 Windows PowerShell 5.1 改寫引號。
    $ValidatorPath = Join-Path $StageDir '_validate_python_syntax.py'
    $ValidatorCode = @'
import ast
import pathlib
import sys
source_path = pathlib.Path(sys.argv[1])
source_text = source_path.read_bytes().decode("utf-8-sig")
ast.parse(source_text, filename=str(source_path))
print("Python syntax OK: " + source_path.name)
'@
    Set-Content -LiteralPath $ValidatorPath -Value $ValidatorCode -Encoding ASCII
    $PythonFiles = @('stock_dashboard.py', 'industry_rotation.py', 'us_industry_indicators.py', 'run_industry_rotation_update.py')
    foreach ($Name in $PythonFiles) {
        $Path = Join-Path $StageDir $Name
        & $PythonExe $ValidatorPath $Path
        if ($LASTEXITCODE -ne 0) { throw "Python 語法檢查失敗：$Name；本地正式檔案尚未覆蓋。" }
    }

    # 額外檢查 27 個產業主題及每群代表股分類，不讓不完整的分類表覆蓋正式檔案。
    $CategoryValidatorPath = Join-Path $StageDir '_validate_rotation_categories.py'
    $CategoryValidatorCode = @'
import ast
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
themes = None
labels = None
for node in tree.body:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id == "ROTATION_THEMES":
            themes = ast.literal_eval(node.value)
        elif node.target.id == "ROTATION_DISPLAY_NAMES":
            labels = ast.literal_eval(node.value)
if not isinstance(themes, dict) or len(themes) != 27:
    raise SystemExit(f"Expected 27 ROTATION_THEMES groups, got {len(themes) if isinstance(themes, dict) else 'missing'}")
if not isinstance(labels, dict) or not set(themes).issubset(labels):
    raise SystemExit("ROTATION_DISPLAY_NAMES must include all 27 groups")
for group, symbols in themes.items():
    if not isinstance(symbols, list) or not symbols or any(not str(symbol).strip() for symbol in symbols):
        raise SystemExit(f"Invalid constituent list: {group}")
    if len(symbols) != len(set(map(str, symbols))):
        raise SystemExit(f"Duplicate constituent in group: {group}")
print(f"Sector mapping OK: {len(themes)} groups; {sum(map(len, themes.values()))} group-stock assignments.")
'@
    Set-Content -LiteralPath $CategoryValidatorPath -Value $CategoryValidatorCode -Encoding ASCII
    & $PythonExe $CategoryValidatorPath (Join-Path $StageDir 'industry_rotation.py')
    if ($LASTEXITCODE -ne 0) { throw '產業分類檢查失敗；本地正式檔案尚未覆蓋。' }

    # 所有檔案下載與檢查均通過後才備份並更新，避免半套版本。
    foreach ($File in $Files) {
        $Target = Join-Path $ProjectRoot $File.Name
        $StagePath = Join-Path $StageDir $File.Name
        if (Test-Path -LiteralPath $Target) {
            Copy-Item -LiteralPath $Target -Destination (Join-Path $BackupDir $File.Name) -Force
        }
    }
    foreach ($File in $Files) {
        Copy-Item -LiteralPath (Join-Path $StageDir $File.Name) -Destination (Join-Path $ProjectRoot $File.Name) -Force
    }

    Write-Host "`n本地程式已更新，原始檔案備份在：$BackupDir" -ForegroundColor Green

    $UserToken = [Environment]::GetEnvironmentVariable('GITHUB_TOKEN', 'User')
    if (!$env:GITHUB_TOKEN -and $UserToken) { $env:GITHUB_TOKEN = $UserToken }
    if (!$env:GITHUB_TOKEN) {
        Write-Warning '目前環境找不到 GITHUB_TOKEN。排程可建立，但無法將每日 JSON 發布到 GitHub；請先設定使用者環境變數。'
    }

    $InstallTask = Read-Host '是否建立／更新每週一至週五 15:10 的 Windows 排程？(Y/N)'
    if ($InstallTask -match '^(Y|YES)$') {
        $SetupPath = Join-Path $ProjectRoot $TaskSetupName
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SetupPath
        if ($LASTEXITCODE -ne 0) { throw '程式已更新，但建立排程失敗；請查看上方錯誤訊息。' }
    } else {
        Write-Host '已略過排程設定；之後可手動執行 setup_industry_rotation_task.ps1。' -ForegroundColor Yellow
    }

    $RunInitial = Read-Host '是否現在先更新一次最新可用交易日資料並發布到 GitHub？初次下載可能需要數分鐘。(Y/N)'
    if ($RunInitial -match '^(Y|YES)$') {
        if (!$env:GITHUB_TOKEN) { throw '未設定 GITHUB_TOKEN，無法將初始資料發布到 GitHub。請先設定後重新執行。' }
        & $PythonExe (Join-Path $ProjectRoot 'run_industry_rotation_update.py') --force
        if ($LASTEXITCODE -ne 0) { throw '初始資料更新未成功；請查看 logs\industry_rotation_task.log。程式檔與備份仍在。' }
    }

    Write-Host "`n更新流程結束。請查看 logs\industry_rotation_task.log，確認每日資料日期及 GitHub 發布結果。" -ForegroundColor Green
} catch {
    Write-Host "`n更新未完全成功：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "原有程式備份（若已完成備份）：$BackupDir" -ForegroundColor Yellow
    throw
} finally {
    if (Test-Path -LiteralPath $StageDir) {
        Remove-Item -LiteralPath $StageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
