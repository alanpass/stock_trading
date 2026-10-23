# 盤後研究參考資料

## 工具對照（MCP 與命令列）

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
| 技術面型態（選用） | `analyze_entry(symbol)`：5 分 K／日 K 型態與日線趨勢 | `python entry_analysis.py analyze <代號>` |
| 研究包外補查 | `research_search_live_news` | `... call search_live_news "{\"query\":\"...\"}"` |
| **品質檢查** | `validate_afterclose_notes` | `python -c "import json,quality_checks as q;print(json.dumps(q.validate_afterclose('<DATE>'),ensure_ascii=False,indent=1))"` |

## agent_notes.json 完整格式

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
