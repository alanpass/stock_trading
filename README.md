# AI 台股即時互動式分析系統 v55

## 本版新增

1. 修正「目前股票相關消息」誤顯示「目前沒有選取股票」：統一使用 `st.session_state.selected` 作為全站唯一選取標的，並寫入 `data/last_selected_symbol.json`。
2. 新增「今日漲跌原因」盤後研究頁籤，針對目前選取股票整合：
   - 今日實際價格/漲跌幅
   - TAIEX 相對表現
   - 同細分業務群相對表現
   - 外資/投信/自營商/三大法人近10日交易
   - 公司主要業務/產業鏈
   - 官方重大訊息
   - 財報/營收
   - 董事長/總經理/法說會/展望/接單等公開談話
   - 產業供應鏈與主要產品需求
   - 中美/關稅/伊朗/油價/利率/AI 等總體與國際事件
3. 每則消息做結構化查證：業務相關度、跨來源數、官方匹配、查證狀態。
4. 新增 `post_market_analysis.py`：將「今天為什麼上漲/下跌」拆成候選驅動因素，不把相關性直接當成因果。
5. Qwen3 8B 僅統整證據與提出研究結論，不擅自修改 production model。
6. 每日盤後自動研究：Windows 任務可從 `data/last_selected_symbol.json` 取得最後選取股票，研究報告加入 `post_market_selected`。
7. 修正進場建議盤後價格：若 Quote 尚未同步今天，優先使用今天最後一根 5 分鐘 K 作為目前價格，避免退回昨天價格。

## 官方資料設計

- TPEx 產業價值鏈：公司個體的主要業務與產業鏈位置，用於建立公司專屬研究 query。https://ic.tpex.org.tw/
- TWSE / TPEx 官方 OpenAPI：重大訊息、月營收、財務資料、法人/盤後資料等。
- DGBAS / MOF / CBC 等官方總經資訊可作為國際/總體事件查證的一級或二級證據。

## 本機 Agent

- Ollama
- Qwen3 8B
- 完全不使用 OpenAI API

## 啟動

```powershell
cd C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard
python -m pip install -r requirements.txt
python -m streamlit run stock_dashboard.py --server.address 127.0.0.1 --server.port 8501
```


## 16. 利多題材雷達

盤後研究 Agent 會依公司主要業務與產業鏈，把新聞、官方佐證、營收、接單/出貨、同業涵蓋度整合成「利多題材」證據鏈。題材強度只是研究訊號，不是投資評級；若利多敘事缺乏基本面或官方支撐，會標示為待驗證。

Dashboard 的「模型研究 Agent」區新增互動式利多題材圖表與逐題材證據明細。


## v53 → v54 利多題材公司清單

利多題材雷達現在除了顯示題材強度，也會同步列出每個題材涉及的自選股公司、主要業務與產業鏈位置。

例如「AI 伺服器」題材會顯示哪些自選股真正屬於伺服器、電源、PCB、被動元件、連接器、儲存等相關業務，避免只看到產業題材名稱卻不知道實際受惠公司。
## 16. 法人說明會與近期產業漲跌研究（v54）

### 法人說明會

系統新增 `earnings_call_agent.py`，依公司代號抓取公開資訊觀測站（MOPS）法人說明會資料，保存：

- 法說日期／時間／地點
- 法人說明會擇要訊息
- MOPS 原始查詢連結
- 規則式證據分數：偏利多／偏利空／中性／待確認
- 利多與利空關鍵字
- 法說後 1 個交易日、5 個交易日股價反應

分類只代表「法說文字中的營運訊號」，不代表未來股價必然方向。若法說內容偏利多但股價反而下跌，Agent 會標示市場反應與事件文字不一致，交由 Qwen3 進一步研究。

### 最近漲價／下跌個股與產業

系統新增 `market_mover_analysis.py`：

```text
Fugle 全市場 snapshot
        ↓
當日漲跌幅前段候選 + 自選股
        ↓
抓 60 日歷史日K
        ↓
計算 5 日 / 20 日報酬
        ↓
依產業代碼彙整
        ↓
抓少量公司新聞
        ↓
分析可能的需求／價格／成本／庫存／訂單原因
```

Dashboard 會分開顯示：

1. 最近漲幅較明顯的個股
2. 最近跌幅較明顯的個股
3. 各產業今日、5日、20日平均變化
4. 每檔公司的主要業務、產業鏈位置
5. 利多／利空原因候選與證據信心

### 自動排程

原本的 `TaiwanStock_ModelResearch_Agent` 不需要另外建立第二個排程；15:30 盤後研究時會一併執行：

- 模型健康度
- 法說會
- 財報／營收
- 近期市場漲跌與產業分布
- 新聞與公司主要業務交叉分析
- 利多題材雷達
- Qwen3 本機研究摘要

### 重要資料原則

法說會以 MOPS 原始資料為優先；漲跌原因先由新聞與公司業務建立「可能原因」，沒有官方公告、法說或財報驗證時，一律保留「待查證」狀態。



## 17. AI 盤後財報 Email Agent（v55）

本版新增 `email_agent.py`，把每日盤後研究結果自動整理成 HTML Email，並透過 Gmail SMTP 寄送。

### Email 報告內容

- 近期上漲／下跌個股
- 產業今日／5日／20日強弱
- 法人說明會與法說後 1／5 日市場反應
- 利多題材與實際涉及公司
- 財報／營收
- 模型健康度與重訓檢查
- Qwen3 8B 本機研究摘要
- 資料來源與查證狀態

### 自動寄送流程

```text
15:30 Windows Task Scheduler
        ↓
ResearchAgent
        ↓
法說會 + 產業漲跌 + 財報 + 利多題材 + Qwen3
        ↓
research_YYYY-MM-DD.json / .md
        ↓
EmailAgent
        ↓
Gmail SMTP
        ↓
a1113359@mail.nuk.edu.tw
```

### Gmail 設定

請複製 `.env.example` 成 `.env`，再填入寄件 Gmail 與 App Password。不要將 App Password 寫入程式或提交到公開儲存庫。

測試：

```powershell
python email_agent.py --status
python email_agent.py --send-latest
```

Dashboard 也可以手動寄送最新盤後財報。

### 重複寄送保護

Email Agent 會建立：

```text
data/email/sent_YYYY-MM-DD.json
```

同一天的盤後報告預設只寄送一次；需要重寄時可以使用 `--force`。


## v57 更新

- 模型研究 Agent 可在任何日期手動執行，不因週末或休市日鎖住。
- Windows 15:30 盤後研究排程改為每日執行；休市日行情分析會回退到最近成功交易日 snapshot。
- MOPS 法說會改用 mopsov 主入口 + 舊入口 fallback，補齊 queryName/inpuType，並對空快取強制重新抓取。
- 新增 TWSE/TPEx 重大訊息「召開法人說明會」補漏路徑。
- Email 改成「今日先看這裡」優先版，先顯示上漲/下跌個股、產業強弱、近期法說、利多題材與營收重點。
- Qwen3 8B 研究摘要改成精簡研究包，並做輸出年份一致性檢查，避免把舊報告內容誤帶入當日 Email。


## v57 修正

- 非交易日依臺灣證券交易所官方市場開休市日曆判斷；週末與國定休市日仍可手動執行盤後研究。
- Windows 15:30 研究排程改為每日執行，休市日仍建立研究報告。
- MOPS 法說查詢加入最近 6 個月逐月補抓、民國日期緊湊格式解析，並以 TWSE/TPEx 官方重大訊息作法說補漏。
- 法說會去重改為「公司＋法說日期」，避免同一公司新法說被舊紀錄擋掉。
- 6274 台燿補上官方產業價值鏈分類：印刷電路板 > 銅箔基板，避免誤顯示為 ETF/無產業分類。
- Email 盤後摘要維持先結論後明細，若 Qwen 出現研究包外年份則不納入摘要。


## Fugle 法說會備忘錄（v59）

法說會研究現在以富果官方「法說會備忘錄」為主要來源：
https://blog.fugle.tw/topic/earnings-call-memo

每次研究會依序：
1. 開啟法說會備忘錄主題頁。
2. 開啟股票代號 Tag 頁。
3. 找到相關文章後，實際開啟文章詳細頁。
4. 擷取營運摘要、產品/業務、財務表現、展望與 Q&A。
5. 再做利多/利空規則式分類。
6. MOPS/TWSE/TPEx 僅在 Fugle 沒有備忘錄時作為官方補漏，不再把網站導覽文字當成法說摘要。


## v60 法說會 Agent 行為

法說會摘要主來源固定為 Fugle：
https://blog.fugle.tw/topic/earnings-call-memo

Agent 必須：
1. 開啟法說會備忘錄主題頁。
2. 嘗試開啟公司 Tag 頁。
3. 找到文章 URL 後，實際開啟文章詳細頁。
4. 讀取營運摘要、產品、市場、財務、展望、Q&A。
5. 將詳細正文送進本機 Qwen3 8B 作為研究包。

若沒有成功開啟正文，只保留「官方法說日程」，不會把 MOPS 網站導覽文字當法說摘要。


### v60 強制規則

- 法說會 Agent 不得只回傳網址。
- 找到 Fugle 法說會備忘錄 URL 後，必須 HTTP 開啟文章詳細頁並擷取正文。
- 正文成功擷取後，Dashboard 與 Email 都必須先顯示摘要，再提供「原文」連結。
- 若未成功開啟正文，狀態顯示「未取得Fugle備忘錄摘要」，不可把 MOPS 網站導覽文字冒充摘要。
- 可用 `python test_fugle_memo.py 6515` 驗證。


## v61：Fugle 法說會備忘錄 AI Agent 重構

主要來源固定為：
https://blog.fugle.tw/topic/earnings-call-memo

每篇文章都由 Ollama `qwen3:8b` Agent 逐一處理。Agent 必須呼叫 `read_fugle_memo(url)`，該 tool 才會開啟 Fugle 詳細文章並讀取正文；沒有成功讀取正文就不能產生法說摘要。

MOPS/TWSE/TPEx 不再混入 `earnings_calls` 摘要，避免網站導覽文字污染。

每日 15:00 由 Windows Task Scheduler 執行；研究與交易日分離，休市日仍可執行法說／新聞／模型研究。

Email 收件者預設為 `a1113359@mail.nuk.edu.tw`，寄件帳號與 App Password 仍由 `.env` 管理。


## v62 法說會流程保證

- 法說會主來源固定為 https://blog.fugle.tw/topic/earnings-call-memo。
- Agent 必須透過 Ollama Tool Calling 呼叫 `read_fugle_memo(url)`。
- Tool 會實際開啟 `https://blog.fugle.tw/post/earnings-call-*` 詳細文章後再回傳正文。
- MOPS/TWSE/TPEx 不可當作法說摘要來源。
- 報告內顯示 Agent 摘要、財務重點、營運重點、展望、利多、利空/風險與 Q&A。
- `source_url` 只用於原始資料查閱。
- 每日 15:00 執行，非交易日也可以執行研究。


## v62 法說會 Agent 強制工具門檻

法說會摘要成功的唯一條件是：Ollama Agent 已呼叫 `read_fugle_memo(url)`，而 tool 已實際開啟 Fugle 詳細文章正文。若 Agent 沒有完成 tool call，系統不產生偽摘要、不使用 MOPS 導覽、不使用模型記憶代替。

主要來源固定為 `https://blog.fugle.tw/topic/earnings-call-memo`。每日 15:00 由盤後 ResearchAgent 呼叫法說會 Agent；非交易日也可執行研究。Email 收件者預設為 `a1113359@mail.nuk.edu.tw`。


## v63 法說會閱讀器重構

法說會備忘錄抓取採三層讀取：requests → curl_cffi Chrome impersonation → Selenium Chrome headless。
若一般 HTTP 回應是網站錯誤頁或防護頁，會自動改用下一層。
主題頁會在 headless Chrome 中嘗試「載入更多」，再逐一開啟 earnings-call 詳細文章。
只有 `detail_read_verified=true` 的文章才允許 Ollama Agent 產生摘要與利多/利空分析。

本機測試：
`python test_earnings_agent.py 6582 --crawl-only`

應看到 `reader_method`、`memo_chars`、`sections`，確認 Agent 真正讀到文章正文。

### CNYES 近兩日新聞 Agent
盤後研究會先爬取 `https://news.cnyes.com/news/cat/headline` 的近兩個日曆日新聞，保存至 `output/research_reports/cnyes_news_YYYY-MM-DD.json`，再由 Qwen3 8B Agent 使用 `get_cnyes_news_overview`、`search_cnyes_news`、`get_cnyes_news_batch` 自主找原因、做產業鏈關聯與研究摘要。

# v8 三段式研究自動化

- 23:00：`run_cnyes_news_nightly.py` 只負責把當天 CNYES 頭條新聞存成日檔。
- 08:00：`run_morning_report.py` 讀最近兩份 CNYES 日檔＋最近五日 Fugle 法說會 cache，由 Qwen3 產生早報。
- 15:00：`run_after_close_research.py` 啟動快速盤後模式，讀快取、不重新爬 CNYES、不重複跑大量 Google News；Agent 查證輪數預設 6。

早報內容：
1. 最近 5 天法說會
2. 最近 2 天財經報導摘要
3. Agent 觀察產業類別
4. Agent 觀察股票與原因

盤後 Email：
- ① 今日 AI 研究重點
- 正向訊號／負向訊號／待追蹤
- 一、今日市場重點（漲跌表以「產業／業務」取代「原因候選」）
- 二、最近 5 天法說會
- ④ 目前主要利多題材
- ⑤ 營收成長重點
- 不顯示③ 法說會訊號、不顯示模型健康度。


## v9 網站與排程調整

Dashboard 最後三項固定為：AI 隔日預測、 市場情報與利多／利空分析、 模型研究 Agent。

Windows 排程：08:30 早報、14:30 盤後研究、23:00 CNYES 當日新聞爬蟲。三個 Task 都保留 WakeToRun，但移除 StartWhenAvailable，因此 Windows 開機／登入不會因錯過時間而立即補跑。

新增 `streamlit_app.py`、`.streamlit/config.toml`、`.gitignore` 與 `DEPLOY_STREAMLIT_CLOUD.md`，用於 Streamlit Community Cloud 部署。
