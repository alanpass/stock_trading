---
name: pipeline-health
description: 台股研究流程的維運 Agent。檢查早報、盤後研究、財經新聞、法說會、網站發布與 Windows 排程的狀態，診斷問題並自動修復可安全重跑的步驟；會寄信的步驟不自動重跑。由排程任務每天 10:30 與 20:30 觸發，也可在使用者問「今天流程正常嗎」時執行。
---

# 研究流程健康檢查

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。

## 1. 取得狀態

呼叫 MCP 工具 `pipeline_status`（沒有 MCP 時：`python pipeline_status.py`）。重點看 `problems`、各流程的 `stage` 與 `log_tail`，以及 `windows_jobs`。

## 2. 診斷與處理

逐一處理 `problems`。先讀相關 log 找出真正原因（`output/scheduler_logs/` 下的 `<job>_<日期>.log`、`after_close_research_<日期>.log`、`morning_entry_<日期>.log`），再依下表行動：

| 問題 | 可以自動做 | 不可以自動做 |
|---|---|---|
| 財經新聞超過 8 小時沒更新／stale | `python scheduled_job_runner.py finance`，再檢查一次 | — |
| 網站財經資料比本機舊 | `python publish_research_data.py` | — |
| AI 今日重點不是 Claude 寫的 | 依 `finance-news-digest` skill 執行 | — |
| 有法說會等待 Claude（`earnings_pending_claude` > 0） | 依 `earnings-call-memo` skill 執行 | — |
| 個股筆記沒有發布到 GitHub | 讀 `output/research_reports/stock_notes_latest.json` 確認內容後，`python -c "from github_publish import publish_json;from pathlib import Path;p=Path('output/research_reports/stock_notes_latest.json');print(publish_json('output/research_reports/stock_notes_public.json',p.read_text(encoding='utf-8'),'data: republish stock notes'))"` | — |
| 盤後研究包建立失敗（`error_no_market_data`、Windows `child_failed`） | 交易日 18:00 前：`python run_after_close_research.py --prepare` | — |
| 盤後研究或早報 `awaiting_claude`／`notes_written` 但沒寄出 | **不要自動執行 --finalize**（避免重複寄信或晚寄）；在回報中列出，交給使用者決定 | 自動寄信 |
| Email 寄送失敗（`error_email`、`email_failed`） | 讀狀態檔與 log 找原因（SMTP 驗證、網路） | 自動重寄 |
| Windows 排程狀態停在 `running` 超過 2 小時 | 檢查是否還有 python 程序在跑：`powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue \| Select-Object Id,StartTime,Path"` | 結束任何程序 |

每次修復後再呼叫一次 `pipeline_status` 確認問題消失；同一個步驟最多重試 1 次。
不要修改程式碼、`.env`、排程設定；需要改程式才能修好的問題，寫進回報即可。

## 3. 回報

用條列回報：
- 整體：健康／有問題（幾項）
- 每個流程一行：早報、盤後研究、財經新聞、法說會、網站發布 → 狀態與時間
- 已自動修復的項目（做了什麼、結果）
- 需要使用者處理的項目（原因與建議做法）
