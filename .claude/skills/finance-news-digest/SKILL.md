---
name: finance-news-digest
description: 財經資訊頁的「AI 今日重點」與逐篇新聞重點。讀取最新一次新聞爬蟲結果，整理今日市場焦點、追蹤事項與風險，並為重要新聞寫重點，通過品質檢查後寫回網站資料並發布。由排程任務在新聞更新之後觸發，也可手動執行。
---

# 財經資訊 AI 今日重點

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。

新聞爬蟲與規則摘要由 Windows 工作排程器的 `finance` 工作自動完成；這個 Skill 在上面加一層 AI 重點。

## 工具

| 用途 | MCP 工具（tw-stock-research） | 命令列 |
|---|---|---|
| 流程狀態 | `pipeline_status` | `python pipeline_status.py` |
| 新聞清單 | `get_job_brief(job="finance")`（先執行 export） | 讀 `output/claude_jobs/finance/brief.md` |
| **品質檢查** | `validate_finance_digest` | `python -c "import json,quality_checks as q;print(json.dumps(q.validate_finance_digest('output/claude_jobs/finance/digest_notes.json'),ensure_ascii=False,indent=1))"` |

## 1. 匯出最新新聞

先看 `pipeline_status` 的 `finance.age_minutes`：超過 180 分鐘，先執行 `python run_finance_info_update.py --no-earnings`。然後：

```
python finance_digest.py export
```

讀 brief：每則新聞都有利多／利空判斷、產業、規則重點和網址；最後一段是「待撰寫逐篇 AI 重點」。

## 2. 撰寫 `output/claude_jobs/finance/digest_notes.json`

```json
{
  "news": {
    "headline": "60 字內：今天市場的主要焦點",
    "key_points": ["5～8 條最重要的事件，每條 50 字內，寫出影響哪個產業"],
    "watch_items": ["3～5 條接下來要追蹤的事項"],
    "risks": ["最多 3 條風險"]
  },
  "earnings": {"headline": "", "key_points": [], "watch_items": [], "risks": []},
  "articles": [
    {"id": "brief.md 裡 id= 後面的字串（必須完全相同）",
     "points": ["1～3 條重點，每條 60 字內，保留關鍵數字與公司名"],
     "sentiment": "利多|利空|中性|混合", "sectors": ["最多 3 個"], "stocks": ["台股代號"],
     "why": "40 字內：為什麼值得注意", "highlights": ["3～8 個關鍵詞，必須逐字出現在 points 裡"]}
  ]
}
```

沒有法說會資料就省略 `earnings`。「待撰寫逐篇 AI 重點」段落的每一則都要寫 `articles`。
規則：只根據 brief 整理，不編造；多則相關新聞合併成一條因果敘事；不給買賣建議；略過網站廣告或導覽文字（例如「下載APP」）。

## 3. 品質檢查與自我修正（必做）

`validate_finance_digest`：errors 不為 0 就修正後再檢查（最多 3 輪）。

## 4. 套用並發布

```
python finance_digest.py apply output/claude_jobs/finance/digest_notes.json
```

輸出 `APPLIED news=claude ... articles=<則數>` 且發布結果 `"ok": true` 就完成。apply 會再跑一次品質檢查（exit 6 表示有 errors）。發布失敗時回報原因；本機網站仍會顯示新的重點。

## 5. 回報

一行：headline、重點條數、逐篇新聞則數、是否發布成功。
