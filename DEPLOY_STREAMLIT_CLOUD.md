# AI 台股即時互動式分析系統｜網站部署說明

## 1. 建議部署方式

本專案已新增 `streamlit_app.py` 作為 Streamlit Community Cloud entry point。Community Cloud 可從 GitHub repository 部署 Streamlit app，並可在 Advanced settings 設定 Secrets 與 Python 版本。官方文件目前說明 Community Cloud 預設使用 Python 3.12，也支援仍在安全支援期的 Python 版本。

## 2. 「只有知道連結的人能使用」的限制

Streamlit Community Cloud 目前沒有真正的「unlisted / 只有知道網址才能看」權限模式。官方設定是：

- Public and searchable：公開且可被搜尋。
- Only specific people can view this app：私人，需要授權的使用者登入。

因此若資料不希望被搜尋到，請在 App Settings -> Sharing 設成 Private，並只加入需要的人。若只想做「網址＋密碼」形式，也可以另外加入應用程式密碼，但那是應用層保護，不等同於 Cloud 的 private app。

## 3. GitHub 準備

建議使用 private GitHub repository。不要提交 `.env` 或 `.streamlit/secrets.toml`。Fugle API Key 請放到 Streamlit Cloud App Settings -> Secrets；官方也建議不要把未加密 secrets 直接存進 Git repository。

Secrets 範例：

```toml
FUGLE_API_KEY = "你的 Fugle API Key"
```

## 4. 部署

1. 將本專案放入 GitHub repository。
2. 前往 `https://share.streamlit.io/`。
3. Create app。
4. Repository 選你的專案。
5. Branch 選 `main`。
6. Main file path 填 `streamlit_app.py`。
7. Advanced settings -> Python version 建議選 3.12。
8. Secrets 放入 `FUGLE_API_KEY`。
9. Deploy。
10. 到 App Settings -> Sharing 設定 Private 並加入指定 Viewer，或依需求公開。

## 5. 本機 AI 與網站的邊界

目前專案的 Qwen3 8B 使用本機 Ollama `127.0.0.1:11434`。因此「部署到 Streamlit Community Cloud」並不會自動把你的本機 Ollama 搬到雲端；網站端若要即時執行 Qwen3 Agent，需要另外提供雲端可連線的 AI endpoint。

目前建議：

- Streamlit Cloud：負責網站／Dashboard。
- Windows 電腦：繼續負責 08:30 早報、14:30 盤後研究、23:00 CNYES 爬蟲與本機 Qwen3。

兩者先分開，可以降低部署複雜度。

## 6. Dashboard 順序

頁面最後三項現在固定為：

1. AI 隔日預測（優先任務）
2. 市場情報與利多 / 利空分析
3. 模型研究 Agent

前面的區域先顯示自選股、即時行情、趨勢、K 線、進場分析、持股、成交與三大法人。

## 7. Windows 排程

Windows 自動排程（台灣時間）：

- 08:10、11:00、13:30、16:00、18:00、23:00：財經新聞／法說會更新、AI 摘要、驗證資料，成功後自動發布 `finance_info_public.json` 到 GitHub 的 `finance-data` 分支。
- 08:30：AI 台股早報。
- 14:30：AI 台股盤後研究。

財經排程統一執行 `python scheduled_job_runner.py finance`。Runner 只有在本機資料驗證成功、且 GitHub 發布成功後，才會將該次工作記錄為成功。網站財經頁每 60 秒 rerun 一次，資料快取 180 秒；有新發布資料時，開啟中的網站通常會在約 3 分鐘內更新。

原本的喚醒規則仍保留：使用 `WakeToRun`，不啟用 `StartWhenAvailable`，不新增登入時立即補跑的觸發器。因此電腦睡眠時可嘗試喚醒；完全關機時不會執行，錯過的排程也不會在登入後立即補跑。
## 8. 建立排程

PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_research_task.ps1
```

檢查：

```powershell
.\check_research_task.cmd
```

## 9. Windows → GitHub → 網站自動更新（新增）

本專案現在加入 `publish_research_data.py` 與 `GITHUB_AUTO_PUBLISH.md`。

Windows 仍然負責：

- CNYES 新聞爬蟲
- Fugle 法說會
- Ollama / Qwen3 8B 摘要
- 產生完整 `finance_info_latest.json`

完成後會再產生：

```text
output/research_reports/finance_info_public.json
```

這是給網站使用的精簡版，只保留標題、原文網址、AI 摘要、AI 重點、情緒、產業、個股、重要度、時間等，不發布完整新聞／法說會正文。

`scheduled_job_runner.py finance` 在本次財經資訊驗證成功後會自動執行：

```text
python publish_research_data.py
```

由 Git 將 `finance_info_public.json` commit + push 到 GitHub。

### 一次性設定

```powershell
cd "$HOME\OneDrive\Desktop\stock_trading\stock_dashboard"
git remote -v
git config --global credential.helper manager
powershell -ExecutionPolicy Bypass -File .\setup_github_auto_publish.ps1
```

如果 `origin` 尚未存在：

```powershell
git remote add origin https://github.com/你的帳號/你的Repository.git
```

完成 GitHub 認證後測試：

```powershell
python run_finance_info_update.py --quick
python publish_research_data.py --dry-run
python publish_research_data.py
```

### 正式排程

確認 Task Scheduler 最後執行的是：

```powershell
python scheduled_job_runner.py finance
```

而不是直接執行 `run_finance_info_update.py`。這樣才能在研究完成後自動發布。

### 網站端

`stock_dashboard.py` 已改為優先讀：

```text
output/research_reports/finance_info_public.json
```

Cloud 上不需要 Ollama，也不需要本機 CNYES cache。

### 暫時關閉自動發布

```powershell
$env:AUTO_GIT_PUBLISH = "false"
```

不設定時預設為 `true`。
