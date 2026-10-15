---
name: morning-report
description: 台股早報。早上建立研究包，閱讀近兩日鉅亨新聞、近期法說會與自選股資料，撰寫早報 JSON，通過品質檢查後寄給所有訂閱者。由排程任務在 08:40 觸發，也可手動執行。
---

# 台股早報

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。
日期一律用台灣時間 `YYYY-MM-DD`，以下以 `<DATE>` 表示。早報每天都寄（與原本的排程相同，週末也寄）。

## 工具

優先使用 MCP server `tw-stock-research`；沒有時用命令列。研究包相關工具都加上 `job="morning"`（命令列加 `--job morning`）。

| 用途 | MCP 工具 | 命令列 |
|---|---|---|
| 流程狀態 | `pipeline_status` | `python pipeline_status.py` |
| 早報指示＋資料 | `get_job_brief(job="morning")` | 讀 `output/claude_jobs/morning_<DATE>/brief.md` |
| 新聞瀏覽／搜尋 | `research_news_batch`、`research_search_news` | `python claude_tools.py call search_cnyes_news "{\"query\":\"...\"}" --job morning` |
| 公司業務 | `research_company_profile` | `python claude_tools.py call get_company_research "{\"symbol\":\"2408\"}" --job morning` |
| **品質檢查** | `validate_morning_notes` | `python -c "import json,quality_checks as q;print(json.dumps(q.validate_morning('<DATE>'),ensure_ascii=False,indent=1))"` |

## 步驟

### 1. 確認研究包

`pipeline_status` 的 `morning.stage` 是 `sent` → 今天已寄出，直接結束，**不要重複寄信**。
`morning.package` 為 true（Windows 排程 08:30 已建立）→ 直接使用；否則執行 `python run_morning_report.py --prepare`，輸出 `PREPARED=` 才繼續。

### 2. 閱讀並撰寫早報

讀 brief：裡面是**完整的早報任務說明、輸出 JSON 格式**，以及新聞候選、近 5 天法說會、自選股公司業務、前一日研究線索。照著說明研究，需要時用新聞工具深讀原文。

把結果寫入 `output/claude_jobs/morning_<DATE>/morning_notes.json`（UTF-8），欄位與 brief 指定的 JSON 相同：`overview`、`news_summary`、`recommended_industries`、`recommended_stocks`、`watch_items`，另加 `"generated_at": "<ISO 時間>"`。

規則：
- `news_summary` 的 `source_links` 只能放研究包裡真的有的網址，`published_at` 保留原始發布時間。
- `recommended_stocks` 只能從自選股挑；這是「研究觀察」，不寫目標價、不寫買進／賣出。
- 單一來源的敘事標示「待查證」。公司寫成「名稱（代號）」。新聞只摘要重點並附連結，不轉載全文。

### 3. 品質檢查與自我修正（必做）

`validate_morning_notes(report_date="<DATE>")`：errors 不為 0 就修正後再檢查（最多 3 輪）；warnings 逐條判斷是否需要修正。

### 4. 測試與寄送

```
python run_morning_report.py --finalize --dry-run
python run_morning_report.py --finalize
```

`DRY_RUN_OK` 才執行正式版（寄給所有訂閱者）。exit 3：Email 失敗，讀 `output/scheduler_logs/morning_entry_status_<DATE>.json` 回報原因，**不要重複寄信**；exit 4／5／6：依輸出修正後再 finalize。

### 5. 回報

3 行：早報重點、新聞摘要／觀察股票數量、品質檢查結果、Email 是否寄出。
