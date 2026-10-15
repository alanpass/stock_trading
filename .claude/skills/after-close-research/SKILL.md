---
name: after-close-research
description: 台股盤後研究。交易日收盤後建立研究包、閱讀證據、撰寫盤後研究筆記與自選股／熱門股個股筆記，通過品質檢查後寄給訂閱者並發布到網站。由排程任務在交易日 15:45 觸發，也可手動執行。
---

# 台股盤後研究

你是這個專案的盤後研究 Agent。Python 負責抓資料、計算、寄信和發布；你負責閱讀證據、判斷今天該研究什麼、交叉查證，最後寫成研究筆記。

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。
日期一律用台灣時間 `YYYY-MM-DD`，以下以 `<DATE>` 表示。

## 工具

優先使用 MCP server `tw-stock-research` 的工具；這個 session 沒有該 MCP 時，改用右欄的命令列。

| 用途 | MCP 工具 | 命令列 |
|---|---|---|
| 流程狀態 | `pipeline_status` | `python pipeline_status.py` |
| 研究包摘要 | `get_job_brief(job="afterclose")` | 讀 `output/claude_jobs/afterclose_<DATE>/brief.md` |
| 市場強弱 | `research_market_overview` | `python claude_tools.py call get_market_overview "{}"` |
| 新聞瀏覽／搜尋 | `research_news_batch`、`research_search_news` | `python claude_tools.py call get_cnyes_news_batch "{\"offset\":0,\"limit\":50}"`、`... call search_cnyes_news "{\"query\":\"...\"}"` |
| 公司財務／業務 | `research_company_financial`、`research_company_profile` | `... call get_company_financial "{\"symbol\":\"2330\"}"`、`... call get_company_research ...` |
| 法說會 | `research_earnings_events`、`research_read_fugle_memo` | `... call get_earnings_events "{\"symbol\":\"\"}"` |
| 題材 | `research_theme_evidence` | `... call get_theme_evidence "{\"theme\":\"...\"}"` |
| 上一期筆記 | `research_previous_notes` | `... call get_previous_notes "{}"` |
| 三大法人／即時行情 | `get_institutional_flow`、`get_quote`、`get_price_history` | —（研究包已有當日行情） |
| 研究包外補查 | `research_search_live_news` | `... call search_live_news "{\"query\":\"...\"}"` |
| **品質檢查** | `validate_afterclose_notes` | `python -c "import json,quality_checks as q;print(json.dumps(q.validate_afterclose('<DATE>'),ensure_ascii=False,indent=1))"` |

## 步驟

### 1. 確認研究包

用 `pipeline_status` 看 `afterclose`：

- `stage` 是 `skipped`（非交易日）→ 直接結束。
- `stage` 是 `sent` → 今天已寄出，直接結束，**不要重複寄信**。
- `package` 為 true → 進入步驟 2（Windows 排程 14:30 已建好研究包）。
- 否則執行 `python run_after_close_research.py --prepare`（官方行情尚未更新時會每 10 分鐘自動重試，最長約 40 分鐘）。若輸出「尚未到 14:30」或沒有 `PREPARED=`，回報 log 尾端的原因後結束，不要編造資料。

### 2. 研究（自主查證）

先讀 brief，再讀上一期筆記：上一期的假設是「待驗證」，本期要寫出哪些被證實、哪些被推翻。

研究原則：
- 每個重要結論都回答：發生什麼、為什麼、證據是什麼、還缺什麼。
- 新聞要實際讀過（至少瀏覽一輪 `research_news_batch`，再針對要解釋的異動用 `research_search_news` 深讀）。
- 產業異動要追到供應鏈哪一段（例如電子零組件 → PCB／IC 載板／ABF／CCL／高速材料）。
- 只有單一媒體來源時降低 confidence，並在 follow_up 寫查證事項；研究包證據不足才用 `research_search_live_news`。
- 股價預測數字只能引用研究包裡模型的結果；不可以自行推估價格或漲跌機率。
- 不給買賣建議（不寫「建議買進／賣出／加碼」等）。

**多 Agent 分工（可用 Agent／Task 工具時）**：市場總覽與研究筆記由你自己寫；自選股 20 多檔可以分成 2～3 組，各交給一個 subagent 並行查證。給 subagent 的指示要包含：日期、該組股票代號、上面的研究原則、可用的工具，並要求它只回傳該組的 `stock_notes` JSON 陣列。收回後由你合併、統一語氣，並檢查有沒有漏股票。

### 3. 撰寫 `output/claude_jobs/afterclose_<DATE>/agent_notes.json`（UTF-8）

```json
{
  "generated_at": "<ISO 時間>",
  "overview": "今日盤後總結，300～800 字",
  "key_takeaways": ["重點；包含上一期假設的驗證結果"],
  "research_notes": [
    {"importance": "high|medium|low",
     "type": "market|financial|earnings|theme|risk|contradiction|industry|event|data_quality|other",
     "symbol": "2330", "title": "標題", "note": "發生什麼、為什麼、證據、還缺什麼",
     "evidence": ["具體證據，含數字與來源"], "tags": ["標籤"], "confidence": 0.7,
     "follow_up": ["後續查證事項"], "source_links": ["https://..."]}
  ],
  "positive_signals": ["名稱（代號）＋原因"],
  "negative_signals": ["名稱（代號）＋原因"],
  "watch_items": ["後續觀察事項"],
  "earnings_digest": "近期法說會重點",
  "financial_focus": ["名稱（代號）：營收 YoY、EPS 重點"],
  "cnyes_news_digest": ["新聞研究重點"],
  "cnyes_research_digest": {"overview": "", "key_findings": [{"title": "", "summary": "", "related_symbols": [], "source_links": []}], "market_drivers": [], "watch_topics": []},
  "agent_actions": [{"tool": "工具名稱", "reason": "為什麼查", "result": "查到什麼（一句話）"}],
  "stock_notes": [
    {"symbol": "2330", "name": "台積電", "category": "watchlist|hot",
     "headline": "一句話重點", "summary": "150～400 字：今日表現、原因、基本面與題材脈絡",
     "positives": [], "risks": [], "watch": [], "evidence": [], "source_links": [], "confidence": 0.6}
  ]
}
```

`stock_notes`：每一檔自選股都要寫（`category: watchlist`）；另外挑 8～10 檔當日熱門股（`category: hot`，不與自選股重複）。資料不足也要寫，明確標示「資料不足／待查證」並調低 confidence。公司寫成「名稱（代號）」；新聞只引用標題與連結，不轉載全文。寫今日漲跌幅時照研究包的數字（例如「今日上漲 4.67%」）。

### 4. 品質檢查與自我修正（必做）

呼叫 `validate_afterclose_notes(report_date="<DATE>")`：

- `errors` 不是空的 → 逐條修正 `agent_notes.json`，再檢查一次；重複到 `errors` 為 0。
- `warnings` → 逐條判斷：數字或連結確實有誤就修正；確認無誤（例如提到的是前一日漲停）可以保留。
- 最多修正 3 輪；仍有 errors 就停下來回報，不要寄信。

### 5. 測試與寄送

```
python run_after_close_research.py --finalize --dry-run
python run_after_close_research.py --finalize
```

`--dry-run` 輸出 `DRY_RUN_OK` 才執行正式版（寄給所有訂閱者、個股筆記發布到網站）。finalize 會再跑一次品質檢查：

- exit 0：完成。
- exit 3：Email 失敗 → 讀 `output/scheduler_logs/after_close_status_<DATE>.json` 回報原因，**不要重複寄信**。
- exit 5／6：筆記檔或品質檢查有問題 → 依輸出修正後再 finalize。

### 6. 回報

3～5 行：今日研究重點、研究筆記數、個股筆記數、品質檢查結果（修正了幾項）、Email 是否寄出、網站是否發布成功。
