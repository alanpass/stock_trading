# AI 台股盤後研究助理 Agent

### 這一版的核心

這不是把固定欄位交給語言模型改寫，而是讓 Qwen3 8B 自主選擇研究工具、深入查證、發現矛盾、建立研究筆記與記號，再交給 EmailAgent 呈現。

這一版的核心不是「程式固定挑出前 3 名」，而是讓本機 Qwen3 8B 像一名研究助理一樣自主閱讀研究證據。

## Agent 流程

ResearchAgent 先收集：

- Fugle 行情與歷史資料
- TWSE / TPEx 官方資料
- MOPS / 財報 / 月營收
- 三大法人
- 公司主要業務與產業鏈
- 近期新聞
- 法說會
- 題材與接單/出貨證據
- 模型健康度

接著交給 `agent_research_assistant.py`。

Qwen3 8B 可以自行選擇：

- 市場資料
- 公司財務
- 法說會
- Fugle 法說會詳細正文
- 新聞
- 題材證據
- 模型健康度
- 公司研究
- 上一期 Agent 筆記
- 必要時的 Google News RSS 補充搜尋

最後產生：

- overview
- key_takeaways
- research_notes
- importance
- type
- tags
- evidence
- confidence
- follow_up
- earnings_digest
- financial_focus
- model_alert

## 重要設計原則

程式只負責「安全與工具邊界」。

例如：不能捏造資料、必須保留證據、不能把未驗證消息寫成事實、不可輸出 chain-of-thought。

但是「今天什麼最重要」不是程式決定，而是 Agent 決定。

## 持續研究筆記

Agent 每次完成研究會把最後筆記保存至：

`output/research_reports/agent_memory.json`

下一次 Agent 可以透過 `get_previous_notes` 讀取上一期筆記，作為變化比較線索；上一期內容必須再用本次證據驗證。

## 法說會

法說會仍維持：

`Qwen3 Agent -> read_fugle_memo(url) -> Fugle 詳細正文 -> 結構化摘要`

最近 5 天由 `EARNINGS_MEMO_LOOKBACK_DAYS` 控制，預設為 5。

## Email

EmailAgent 只負責呈現與寄送 `agent_research_notes`。

它不再自己重新挑選前 3 / 前 5，也不會第二次呼叫 Qwen3 產生摘要。

每執行一次 `send_report()` 都會寄一次，方便測試。

## Windows 排程

`setup_research_task.ps1` 預設每天 14:30。

只需修改：

```powershell
$ScheduleHour = 15
$ScheduleMinute = 0
```

此版本不建立 AtLogOn recovery task，避免 Windows 開機／登入時立即執行研究。

Sleep / Hibernate 使用 `WakeToRun`；完全關機期間 Windows Task Scheduler 不會執行 Python。

## 測試

測試 Agent：

```powershell
python test_agent_research_assistant.py
```

測試最新 Email：

```powershell
python email_agent.py --send-latest
```

查看 SMTP：

```powershell
python email_agent.py --diagnose-smtp
```

建立排程：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_research_task.ps1
```

## 不覆蓋 stock_api.py

本升級包不包含 `stock_api.py`，用來保留你目前已設定的 Fugle API Key 與現有行情邏輯。


## v4 Email 版面與研究流程

Email 正文保留：
- ① 今日 AI 研究重點
- ② 最近 5 天法說會
- ③ Agent 研究筆記
- ④ 目前主要利多題材
- ⑤ 營收成長重點

Agent 遇到未知或證據不足時，會使用研究工具補查；尤其針對產業異常漲跌，會進一步查近期新聞，嘗試把市場異動與供應鏈題材、公司基本面連起來。

模型健康度不再放在每日 Email 正文，完整技術資訊仍保存於 research JSON / Markdown 附件。

## 鉅亨網近兩日新聞研究（v6）

研究流程現在會在盤後先爬取：

`https://news.cnyes.com/news/cat/headline`

預設抓取「今天＋昨天」兩個日曆日可取得的新聞，先保存到：

`output/research_reports/cnyes_news_YYYY-MM-DD.json`

爬蟲會收集文章 URL、標題、分類、發布時間、摘要、正文前段與標籤。一般先用 requests/BeautifulSoup；若頭條頁是動態載入或 requests 拿不到足夠文章，會自動以 Selenium 滾動補抓。

### Agent 如何使用

ResearchOrchestratorAgent 新增三個工具：

- `get_cnyes_news_overview`：查看兩日新聞總量、分類與代表標題。
- `search_cnyes_news`：依產業、公司、題材、事件等關鍵字搜尋已爬新聞；常見產業詞會做同義詞擴展。
- `get_cnyes_news_batch`：需要廣泛閱讀時按批次取得文章。

Agent 被要求優先用鉅亨近兩日新聞解釋「股價／產業異動為什麼發生」，例如看到電子零組件上漲時，會再檢查 PCB、IC 載板、ABF/BT、CCL、高速材料、AI 伺服器等相關新聞，再與財報、營收、法人與官方資料交叉驗證。

新聞本身只作為市場研究線索；沒有足夠證據時 Agent 必須標記待查證，不得把新聞敘事直接當成事實。

### 可調整設定

在 `.env` 加入：

```env
CNYES_NEWS_DAYS=2
CNYES_NEWS_MAX_ARTICLES=600
CNYES_NEWS_MAX_SCROLLS=18
CNYES_NEWS_FORCE_REFRESH=false
```

測試爬蟲與 Agent 工具：

```powershell
python test_cnyes_news_crawler.py
```

強制重新爬取近兩日新聞：

```powershell
python cnyes_news_crawler.py --days 2 --force --max-articles 260
```

## v7：CNYES 夜間採集 × 盤後新聞 Agent

為避免 15:00 盤後研究等待新聞網站，本版將新聞流程拆成兩個獨立階段：

```text
每日 23:00
CNYES Headline
    ↓
run_cnyes_news_nightly.py
    ↓
cnyes_news_YYYY-MM-DD.json

隔日 15:00
ResearchAgent
    ↓
讀最近兩份 CNYES 日檔（不連線爬蟲）
    ↓
CnyesNewsDigestAgent / Qwen3 8B
    ↓
新聞事件、產業驅動、矛盾、待查證
    ↓
主 ResearchOrchestratorAgent
    ↓
整合行情 / 財報 / 法人 / 法說 / 官方資料
    ↓
Email
```

### 新增檔案

- `cnyes_news_agent.py`：盤後新聞摘要與研究 Agent
- `run_cnyes_news_nightly.py`：每天 23:00 的當日新聞採集入口

### 快取原則

`cnyes_news_crawler.py` 的 `crawl(days=1)` 只抓當天新聞並保存當日日檔。
盤後由 `load_recent_cached()` 只讀最近兩份可用日檔，避免盤後再次連線 CNYES。

### Email

Email 新增「鉅亨近兩日新聞｜Agent 研究摘要」，內容不是逐篇新聞，而是 Qwen3 根據快取新聞形成的事件、原因、影響、證據與信心。

### 排程

`setup_research_task.ps1` 現在建立兩個任務：

- `AI_TW_Stock_CNYES_Nightly_News`：23:00
- `AI_TW_Stock_AfterClose_Research`：15:00

修改檔案最上方 `$NewsHour / $NewsMinute` 與 `$ResearchHour / $ResearchMinute` 即可調整。

## v8 三階段每日研究流程

本版本正式拆成三個階段，避免 15:00 盤後研究等待新聞爬蟲：

```text
23:00 CNYES Nightly Crawler
    ↓
cnyes_news_YYYY-MM-DD.json
    ↓
08:00 AI Morning Report
    ├─ 最近 5 天法說會
    ├─ 最近 2 天 CNYES 財經報導摘要
    └─ Agent 觀察的產業／股票
    ↓
15:00 AI After-Close Report
    ├─ ① 今日 AI 研究重點
    ├─ 正向／負向／待追蹤
    ├─ 一、今日市場重點
    ├─ 二、最近 5 天法說會
    ├─ ④ 主要利多題材
    └─ ⑤ 營收成長重點
```

### v8 早報

入口：`run_morning_report.py`

預設每日 08:00 寄送，讀取：
- 前兩個可用 CNYES 日檔
- 最近五日 Fugle 法說會 cache；若沒有 cache 才補抓
- 自選股 business profile
- 前一份盤後研究的題材／財報線索

早報 Agent 使用 Qwen3 8B 產生「財經事件摘要、影響產業、觀察產業、觀察股票與原因」。

### v8 盤後快速模式

`run_after_close_research.py` 預設設定：
- `AFTER_CLOSE_FAST_MODE=true`
- `AGENT_MAX_TOOL_ROUNDS=6`
- `EARNINGS_USE_CACHE=true`

盤後不重新爬 CNYES、不重新執行 Google News 大量搜尋、不重複啟動法說會全文分析；改讀早報已完成的新聞摘要與 Fugle memo cache。

盤後漲跌表的「原因候選」改為「產業／業務」，只說明公司屬於 PCB、MLCC、IC 代工、封測、CCL 等哪種業務，不在表格內解釋漲跌原因。

### v8 排程

`setup_research_task.ps1` 會建立：
- `AI_TW_Stock_CNYES_Nightly_News`：23:00
- `AI_TW_Stock_Morning_Report`：08:00
- `AI_TW_Stock_AfterClose_Research`：15:00


## v9 網站與排程調整

Dashboard 最後三項固定為：AI 隔日預測、 市場情報與利多／利空分析、 模型研究 Agent。

Windows 排程：08:30 早報、14:30 盤後研究、23:00 CNYES 當日新聞爬蟲。三個 Task 都保留 WakeToRun，但移除 StartWhenAvailable，因此 Windows 開機／登入不會因錯過時間而立即補跑。

新增 `streamlit_app.py`、`.streamlit/config.toml`、`.gitignore` 與 `DEPLOY_STREAMLIT_CLOUD.md`，用於 Streamlit Community Cloud 部署。
