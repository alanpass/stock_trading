# v66 法說會串接修正

## 已修正

1. `test_earnings_agent.py --crawl-only`
   - 不再只把 Memo 印到終端機。
   - 改用 `read_memo_tool()`，因此會寫入 SQLite Memo cache。
   - 同時輸出 `output/research_reports/fugle_earnings_memo_probe_*.json`。

2. `earnings_call_agent.py`
   - 新增統一的 `enrich_earnings_items()`。
   - 自動補齊 `event_date`、`memo_opened`、`detail_read_verified`、`memo_sections`。
   - 將 Ollama `利多/利空/中性/混合` 正規化成 Dashboard 可讀的標籤。
   - 由 Fugle 歷史日 K 計算法說後第 1 與第 5 個交易日報酬。

3. `research_agent.py`
   - 正式盤後研究取得 Fugle Memo 後，統一補齊法說日期與市場反應欄位。
   - 研究包 `research_YYYY-MM-DD.json` 將直接包含完整 `earnings_calls`。

4. `stock_dashboard.py`
   - 即使最新 `research_YYYY-MM-DD.json` 尚未重新生成，只要存在更新的 Fugle Memo JSON/probe，就會即時合併進目前研究結果。
   - 研究報告以檔案 mtime 自動偵測更新，不必依賴舊的 Streamlit session。
   - 法說 Memo enrichment 使用 session cache，避免每次 2 秒刷新都重新打 Fugle 歷史 API。

## 資料可信規則

只有 `detail_read_verified=true` 且存在 `memo_text` 的 Fugle 詳細文章，才能被當作法說會 Memo 顯示；MOPS/TWSE/TPEx 導覽或日程不會冒充正文摘要。
