---
name: stock-deep-research
description: 單一台股的深度研究。使用者指定股票代號或名稱時，用即時行情、三大法人、籌碼、歷史走勢、新聞與財報資料研究這檔股票，寫成個股研究筆記，通過品質檢查後發布到網站「AI 個股研究筆記 → 深度研究」。使用者說「研究一下 2330」「幫我看南亞科」時使用。
---

# 個股深度研究

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。

## 1. 蒐集證據（MCP server `tw-stock-research`）

依序取得，並記下每一項的日期：

1. `get_market_status`：今天是否交易日、盤中或盤後。
2. `get_quote(symbol)`：最新價格、漲跌、成交量。
3. `get_price_history(symbol, days=120)`：近 5／20／60 日報酬、是否創高、波動。
4. `get_institutional_flow(symbol)`：近 10 日外資／投信／自營商買賣超與連續性。
5. `get_holder_structure(symbol)`：大戶／散戶持股變化。
6. `get_stock_note(symbol)`：網站上現有的筆記（當作上一版，比對有什麼變化）。
7. 新聞與基本面：今天有研究包時用 `research_search_news`、`research_company_financial`、`research_company_profile`（job="afterclose" 或 "morning"）；沒有時用 `research_search_live_news("<名稱> <代號>")`。
8. 有近期法說會時：`research_earnings_events(symbol)`，必要時 `research_read_fugle_memo(url)`。

沒有 MCP 時，至少用 `python claude_tools.py call search_live_news "{\"query\":\"<名稱> <代號>\"}" --job afterclose` 補新聞，並在筆記中說明行情資料不足。

## 2. 研究與撰寫

回答四件事：最近發生什麼（價格、法人、籌碼）、為什麼（新聞、財報、產業、題材）、證據是什麼、還缺什麼。
價格與籌碼數字一律照工具回傳的值，不推估目標價或漲跌機率，不給買賣建議。

寫入 `output/claude_jobs/deep/<代號>_<YYYYMMDD>.json`：

```json
{"symbol": "2330", "name": "台積電",
 "headline": "一句話重點",
 "summary": "200～500 字：價格與籌碼變化、原因、基本面與題材脈絡",
 "positives": [], "risks": [], "watch": [],
 "evidence": ["具體證據，含數字與日期"], "source_links": ["https://..."], "confidence": 0.6}
```

## 3. 自我檢查

寫完後自己核對：每個數字都能對應到步驟 1 的工具結果；`source_links` 都是實際查到的網址；沒有「建議買進／賣出」等用語。

## 4. 發布

```
python stock_notes.py upsert output/claude_jobs/deep/<代號>_<YYYYMMDD>.json
```

輸出 `UPSERTED=[...]` 與 `PUBLISH={"ok": true ...}` 就完成。使用者只想看結果、不要發布時，加上 `--no-publish`。

## 5. 回覆使用者

用 5～8 行整理研究結果（附資料日期），並說明筆記已發布到網站「未來分析 → AI 個股研究筆記 → 深度研究」。
