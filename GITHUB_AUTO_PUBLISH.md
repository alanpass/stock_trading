# AI 台股研究中心｜Windows → GitHub → Streamlit 自動財經資訊發布

## 架構

```text
Windows Task Scheduler
        |
        v
run_finance_info_update.py
        |
        +-- CNYES 新聞
        +-- Fugle 法說會
        +-- Ollama / Qwen3 8B
        |
        v
output/research_reports/finance_info_latest.json
        |
        v
publish_research_data.py
        |
        +-- 轉成精簡版 finance_info_public.json
        +-- 只 git add 公開檔
        +-- git commit
        +-- git push origin <branch>
        |
        v
GitHub repository
        |
        v
Streamlit Community Cloud
        |
        v
網路上的財經資訊頁面
```

## 重要設計

### 本機完整資料不公開

以下檔案留在 Windows 本機：

- `output/research_reports/finance_info_latest.json`
- `output/research_reports/finance_summary_cache.json`
- `output/research_reports/finance_run_history.json`

網站公開的是：

- `output/research_reports/finance_info_public.json`

公開版不包含完整新聞正文與法說會 memo 正文，只保留 Dashboard 真正需要的：標題、原文網址、AI 摘要、AI 重點、情緒、產業、個股、重要度、時間等。

## 一次性設定 GitHub

### 1. 先確認本機專案就是 Git repository

PowerShell：

```powershell
cd "$HOME\OneDrive\Desktop\stock_trading\stock_dashboard"
git status
git remote -v
```

應該看到 `origin` 指向你的 GitHub repository。

若沒有 `origin`：

```powershell
git remote add origin https://github.com/你的帳號/你的Repository.git
```

### 2. 設定 Git Credential Manager

推薦使用 Git Credential Manager，不要把 Personal Access Token 寫在 Python 程式裡。

```powershell
git config --global credential.helper manager
```

接著第一次 `git push` 時依 Git for Windows 的登入流程完成 GitHub 認證。

也可以使用 SSH remote；本發布器同樣可以 `git push`。

### 3. 測試發布器

先執行一次財經研究：

```powershell
python run_finance_info_update.py --quick
```

再測試：

```powershell
python publish_research_data.py --dry-run
```

確認沒有問題後，再真正發布：

```powershell
python publish_research_data.py
```

成功會看到：

```text
✅ GitHub 發布成功。
```

## 自動排程

本專案的 `scheduled_job_runner.py` 已經改成：

```text
finance job
   |
   v
run_finance_info_update.py
   |
   v
驗證 finance_info_latest.json
   |
   v
publish_research_data.py
   |
   v
Git commit + push
```

所以原本 08:10、11:00、13:30、16:00、18:00、23:00 的財經資訊排程，只要 Task Scheduler 執行的是：

```powershell
python scheduled_job_runner.py finance
```

就會自動同步網站資料。

## 暫停自動 Git 發布

只研究、不推 Git：

```powershell
$env:AUTO_GIT_PUBLISH = "false"
python scheduled_job_runner.py finance
```

恢復：

```powershell
$env:AUTO_GIT_PUBLISH = "true"
```

排程正式環境預設 `AUTO_GIT_PUBLISH=true`。

## 失敗行為

- CNYES / Fugle / Qwen3 失敗：原本的財經更新驗證會先判斷是否可用，stale 時不會發布。
- JSON 產生成功、GitHub push 失敗：本機資料保留；排程狀態會記為 `publish_failed`，log 會看到 `GIT_PUBLISH_RETURN_CODE`。
- 發布器只會 stage `finance_info_public.json`，不會把你其他 VS Code 修改一起 commit。
- 發布器發現 index 內已有其他 staged 檔案時會直接停止，避免誤提交。

## Streamlit Cloud

網站端 `stock_dashboard.py` 已改成：

1. 優先讀 `finance_info_public.json`。
2. 若公開檔不存在，本機才 fallback 到 `finance_info_latest.json`。

因此 Streamlit Cloud 不需要 Ollama，也不需要看到 CNYES 爬蟲的完整資料。

在 Streamlit Community Cloud：

- Repository：選你的 GitHub repository
- Branch：選目前自動 push 的 branch，例如 `main`
- Main file：`streamlit_app.py`
- Python：建議 3.12
- Secrets：只放 Dashboard 真正需要的 API Key，例如 `FUGLE_API_KEY`

不要把 `.env` 放進 repository。

## 部署後的資料流

當 Windows 18:00 排程完成：

```text
18:00 CNYES / Fugle
      ↓
18:xx Qwen3 完成摘要
      ↓
finance_info_latest.json
      ↓
finance_info_public.json
      ↓
git commit
      ↓
git push
      ↓
GitHub 更新
      ↓
Streamlit Cloud 偵測 GitHub commit
      ↓
網站顯示最新財經資訊
```

Community Cloud 官方文件：GitHub repository 是 App 的來源；push 到 repository 的更新會反映到部署中的 App。
https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
