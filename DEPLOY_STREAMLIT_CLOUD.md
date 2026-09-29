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

`setup_research_task.ps1` 現在預設：

- 08:30：AI 台股早報
- 14:30：AI 台股盤後研究
- 23:00：CNYES 當日新聞爬蟲

三者都使用 `WakeToRun`，但**不使用 `StartWhenAvailable`**。

因此：

- 電腦睡眠／休眠：Windows 可嘗試在指定時間喚醒並執行。
- 完全關機：指定時間不可能執行 Python。
- 開機／登入：不會因錯過時間而立即補跑。
- 若開機時已錯過當日指定時間：等待下一個每日時間。

這正是「排程記得住，但開機不直接啟動，等到指定時間才開始」的行為。

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
