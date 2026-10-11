# Remove ALL old stock-project scheduled tasks, then register the latest version (v13).
#
# Removes : AI_TW_Stock_*  (all versions, incl. Finance_Update_xxxx / Recovery / Catchup / CNYES)
#           TaiwanStock_ModelResearch_Agent  (the very old 15:30 task, otherwise mails go out twice)
# Keeps   : StockTrading-IndustryAnalysis-1510 (industry rotation task) unless you add -IncludeIndustry
#
# Usage (normal PowerShell, no admin needed):
#   cd <project folder>
#   powershell -ExecutionPolicy Bypass -File .\reset_and_setup_tasks.ps1
#   (add -Yes to skip the confirmation, -IncludeIndustry to also remove the industry task)
param(
    [switch]$Yes,
    [switch]$IncludeIndustry
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Setup = Join-Path $ProjectDir "setup_research_task_v13.ps1"
if (-not (Test-Path -LiteralPath $Setup)) {
    Write-Host "ERROR: setup_research_task_v13.ps1 not found in $ProjectDir" -ForegroundColor Red
    exit 1
}

$all = Get-ScheduledTask -ErrorAction SilentlyContinue
$targets = @($all | Where-Object {
    $_.TaskName -like "AI_TW_Stock_*" -or
    $_.TaskName -eq "TaiwanStock_ModelResearch_Agent" -or
    ($IncludeIndustry -and $_.TaskName -like "StockTrading-IndustryAnalysis*")
})

Write-Host ""
if ($targets.Count -eq 0) {
    Write-Host "No old tasks found."
} else {
    Write-Host "Tasks that will be REMOVED:" -ForegroundColor Yellow
    $targets | ForEach-Object { Write-Host ("  - {0}  [{1}]" -f $_.TaskName, $_.State) }
}
$kept = @($all | Where-Object { $_.TaskName -like "StockTrading-IndustryAnalysis*" -and -not $IncludeIndustry })
if ($kept.Count -gt 0) {
    Write-Host "Kept (not touched): $(($kept | ForEach-Object { $_.TaskName }) -join ', ')"
}
Write-Host ""

if ($targets.Count -gt 0 -and -not $Yes) {
    $answer = Read-Host "Remove them and register the latest tasks? (y/N)"
    if ($answer -notmatch '^(y|yes)$') { Write-Host "Cancelled. Nothing was changed."; exit 0 }
}

foreach ($t in $targets) {
    try {
        Stop-ScheduledTask -TaskName $t.TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $t.TaskName -Confirm:$false
        Write-Host "Removed: $($t.TaskName)"
    } catch {
        Write-Host "FAILED to remove $($t.TaskName): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Registering latest tasks (v13) ..." -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File $Setup
if ($LASTEXITCODE -ne 0) { Write-Host "setup_research_task_v13.ps1 failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "Tasks now registered:" -ForegroundColor Green
Get-ScheduledTask | Where-Object { $_.TaskName -like "AI_TW_Stock_*" } | ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    Write-Host ("  {0,-34} next run: {1}" -f $_.TaskName, $info.NextRunTime)
}
