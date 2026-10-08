# 財經資訊自動化（v12）

## 排程
| 時間 | 工作 | 說明 |
|---|---|---|
| 08:10、11:00、13:30、16:00、18:00、23:00 | `finance` | 爬鉅亨新聞（前一日 00:00～現在）＋法說會 → Agent 摘要／畫重點／分析 → **覆蓋** `finance_info_latest.json` |
| 08:30 | `morning` | 晨報。用 08:10 的快取（超過 90 分鐘才補更新） |
| 14:30 | `afterclose` | 盤後分析報導。用 13:30 的快取 |

安裝／更新排程：`powershell -ExecutionPolicy Bypass -File .\setup_research_task_v12.ps1`
手動測試：
- 快速（約 1 分鐘，只爬新聞＋規則摘要，不用 Ollama、略過法說會）：`python run_finance_info_update.py --quick`
- 完整（會即時顯示進度）：`python scheduled_job_runner.py finance`
- 其他旗標：`--no-llm`（不呼叫 Qwen3）、`--no-earnings`（略過法說會）、`--no-crawl`（只重跑 Agent 分析）

## 每次更新做的事
1. **爬取**：頭條＋台股＋國際股＋外匯＋期貨（`CNYES_CATEGORIES` 可改），沒有的分類自動略過。
2. **取代舊資料**：每次寫入 `cnyes_news_latest.json`、同日日檔、`finance_info_latest.json`（原子覆蓋）。不再產生時間戳快照。
   若這次完全爬不到，保留上一版並標示 `stale`（儀表板會顯示警告，排程記錄為失敗）。
3. **Agent 分析**（`finance_agent.py`）
   - 摘要：每篇 1～3 個重點（`ai_points`）；同時寫回 `summary`，晨報／Email 直接受惠
   - 畫重點：`highlights`（數字、公司、事件詞）→ 儀表板以紅字螢光標示
   - 分析：利多／利空／中性／混合、影響產業、相關個股、重要度（★）、「為什麼重要」
   - 去重：同一事件的重複報導合併
   - 總覽：`news_digest`＝今日重點、利多／利空主題、產業熱度、待追蹤、風險
   - Qwen3 可用時重寫重要度前 40 篇＋總覽；已摘要過的存 `finance_summary_cache.json`，下次不重做

## 可調整的環境變數（.env）
| 變數 | 預設 | 說明 |
|---|---|---|
| `CNYES_CATEGORIES` | headline,tw_stock,wd_stock,forex,future | 要爬的鉅亨分類 |
| `CNYES_NIGHTLY_MAX_ARTICLES` | 600 | 一次最多保留幾篇 |
| `FINANCE_USE_LLM` | true | false = 只用規則層，不呼叫 Ollama |
| `FINANCE_LLM_MAX_ARTICLES` | 40 | 每次最多讓 Qwen3 重寫幾篇 |
| `FINANCE_LLM_TIME_BUDGET_SEC` | 900 | Qwen3 摘要的時間上限 |
| `FINANCE_MIN_NEWS` | 8 | 鉅亨少於這個數量才補 Google News 備援 |
| `EARNINGS_FORCE_REFRESH` | false | true = 法說會全部重新分析 |
| `EARNINGS_NEW_PER_RUN` | 12 | 每次最多新分析幾場法說會，其餘留給下一個時段 |
| `EARNINGS_TIMEOUT_SEC` | 600 | 法說會步驟逾時秒數，逾時就沿用既有資料 |
| `MORNING_NEWS_MAX` / `EMAIL_NEWS_MAX` | 15 / 12 | 晨報／Email 最多列幾則新聞 |
| `MORNING_FINANCE_FRESH_MINUTES` | 90 | 晨報認為快取「夠新」的分鐘數 |

## 「財經新聞只有兩篇」的原因與修正
- `cnyes_news_crawler.py` 的 `crawl()` 使用了 `os.getenv` 卻沒有 `import os`，抓到文章後會 `NameError` 中斷，**快取檔根本寫不出來**。
- 舊排程每次只抓「當天 00:00 到現在」且只抓頭條分類，早上 08:30 的當天視窗只有很少幾篇，還會把當天檔整個覆蓋。
- 診斷：`python diagnose_finance_news.py` 會列出每個快取檔幾篇、每個分類抓幾篇、每個時段的歷史。
- 離線驗證：`python test_finance_pipeline.py`

## 儀表板「財經資訊」頁
今日重點卡片（含利多／利空主題、產業熱度、待追蹤）＋新聞清單（可依情緒／產業篩選、依重要度或時間排序、每次顯示 15 則，可「顯示更多」）。
