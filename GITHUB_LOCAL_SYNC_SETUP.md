# Windows 本機 → GitHub → Streamlit 網站：財經資料同步設定

## 新架構

- `main`：只放網站程式。只有你更新程式碼才觸發 Streamlit 部署。
- `finance-data`：只放 `output/research_reports/finance_info_public.json`，不放本機完整快取。
- Windows 工作排程：先跑原本 CNYES／法說會／Qwen3 更新；資料驗證通過後，使用 GitHub Contents API 更新 `finance-data` 分支。
- Streamlit：每 60 秒讀取 `finance-data` 的公開 JSON。資料來源短暫失效時，才退回本機快取（本機執行才有）。

因此不需要本機 `.git`、`git remote` 或 `git push`，資料更新不會觸發主分支重新部署。

## 1. GitHub Token（只做一次）

在 GitHub 建立 Fine-grained personal access token：

1. Repository access 選 `Only select repositories` → `alanpass/stock_trading`。
2. Repository permissions 將 `Contents` 設為 `Read and write`。
3. 不要把 token 貼到聊天、程式碼或提交到 GitHub。

在 Windows PowerShell 設定給目前 Windows 使用者：

```powershell
[Environment]::SetEnvironmentVariable('GITHUB_TOKEN', '貼上你的Token', 'User')
$env:GITHUB_TOKEN = [Environment]::GetEnvironmentVariable('GITHUB_TOKEN', 'User')
```

請注意：`setx`／User 環境變數會把 token 存在本機使用者環境設定中。不要在共用電腦使用；若想避免長期保存，可只在測試 PowerShell 視窗設定 `$env:GITHUB_TOKEN = '...'`，但關閉視窗後排程不會繼承該值。Token 遺失或外洩時請立刻在 GitHub 撤銷並重新建立。

設定完成後，請關閉並重新開啟 PowerShell，確認只顯示有無設定，不要輸出 token：

```powershell
if ($env:GITHUB_TOKEN) { 'GITHUB_TOKEN 已設定' } else { 'GITHUB_TOKEN 未設定' }
```

## 2. 更新本機專案檔案

先備份目前專案資料夾，再將此補丁中的 `stock_dashboard.py`、`publish_research_data.py`、`scheduled_job_runner.py` 放到本機專案根目錄覆蓋。不要覆蓋你的 `.env`、`data`、`models` 或 `output`。

## 3. 手動先更新，再發布

在專案根目錄執行：

```powershell
python -m py_compile stock_dashboard.py publish_research_data.py scheduled_job_runner.py
python run_finance_info_update.py --quick
python publish_research_data.py
```

`--quick` 是流程測試模式：不呼叫 Qwen3 且略過法說會，適合先驗證資料連線。正式發布請用正常排程或 `python scheduled_job_runner.py finance`。

成功後，檢查這個檔案是否有更新：

`https://github.com/alanpass/stock_trading/blob/finance-data/output/research_reports/finance_info_public.json`

原始公開 JSON 網址：

`https://raw.githubusercontent.com/alanpass/stock_trading/finance-data/output/research_reports/finance_info_public.json`

## 4. 自動排程

原本 Task Scheduler 的財經任務改為／維持執行：

```powershell
python scheduled_job_runner.py finance
```

排程執行帳號必須能讀到 `GITHUB_TOKEN`。如果排程是用不同 Windows 帳號或「不論使用者是否登入都執行」，請確認該帳號的環境變數也已設定；環境變數新增後需重新開啟排程程序或重新登入。

發布 log 在 `output/scheduler_logs/finance_YYYY-MM-DD.log`。請尋找：

- `GITHUB_DATA_PUBLISH=SUCCESS`：已發布。
- `GITHUB_DATA_PUBLISH_RETURN_CODE`：發布器結束碼，`0` 代表成功。
- `PUBLISH_FAILED`：依後續錯誤訊息檢查 Token 權限、分支或網路。

## 5. 網站端驗證

- 網站資料 URL 指向 `finance-data` 分支，不是 `main`。
- 儀表板的「最後更新」顯示的是本機研究結果的 `updated_at`，不是 Streamlit 程式部署時間。
- Streamlit 每 60 秒重新取得 JSON；瀏覽器仍受頁面自動刷新設定影響。
- 如果 GitHub JSON 已更新但網站未更新，等待 60 秒後重新整理頁面。Streamlit Cloud 不需要也不應執行本機 Ollama／爬蟲排程。

## 資料與安全

只發布白名單欄位（標題、摘要、重點、情緒、產業、個股、來源網址與統計），不發布完整新聞／法說會正文或 `.env`。由於 `finance-data` 分支是公開 repository 的分支，請假設其中 JSON 的所有欄位任何人都可讀取。
