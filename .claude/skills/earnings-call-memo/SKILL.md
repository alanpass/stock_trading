---
name: earnings-call-memo
description: Fugle 法說會備忘錄研究。讀取尚未分析的法說會正文，整理財務、營運、展望、利多利空、風險與 Q&A，通過品質檢查後寫回網站資料並發布。由排程任務在每天 18:40（Windows 18:00 法說會更新之後）觸發，也可手動執行。
---

# 法說會備忘錄研究

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。

Fugle 備忘錄的爬取由 Windows 工作排程器的 `earnings` 工作完成，Python 先用規則產生保守重點；這個 Skill 讀正文後寫出真正的研究摘要。

## 工具

| 用途 | MCP 工具（tw-stock-research） | 命令列 |
|---|---|---|
| 待分析數量 | `pipeline_status`（`finance.earnings_pending_claude`） | `python earnings_memo.py export` |
| 正文 | `get_job_brief(job="earnings")`（先執行 export） | 讀 `output/claude_jobs/earnings/brief.md` |
| 補讀全文 | `research_read_fugle_memo(url)` | — |
| **品質檢查** | `validate_earnings_memo` | `python -c "import json,quality_checks as q;print(json.dumps(q.validate_earnings_memo('output/claude_jobs/earnings/memo_notes.json'),ensure_ascii=False,indent=1))"` |

## 1. 匯出待分析的法說會

```
python earnings_memo.py export
```

`PENDING=0` → 直接結束。否則讀 brief，每場都有 `url` 與備忘錄正文（依段落分開）；正文被截斷時用 `research_read_fugle_memo` 讀全文。

## 2. 逐場分析，寫入 `output/claude_jobs/earnings/memo_notes.json`

```json
{
  "items": [
    {"url": "brief 裡該場的 url（必須完全相同）",
     "impact": "利多|利空|中性|混合", "confidence": 80,
     "one_line_summary": "一句話總結，含最重要的數字",
     "key_points": ["財務：…", "營運：…", "展望：…", "風險：…"],
     "highlights": ["關鍵數字或關鍵字"],
     "financial_highlights": ["營收、毛利率、EPS 等，含年增／季增"],
     "operating_highlights": ["產品線、客戶、產能、接單"],
     "guidance": ["公司對下一季／全年的說法"],
     "positive_factors": [], "negative_factors": [], "key_risks": [],
     "qa_highlights": ["Q&A 中有資訊量的問答"],
     "reasoning": "為什麼判斷為利多／利空／混合"}
  ],
  "digest": {"headline": "這批法說會的共同趨勢（60 字內）", "key_points": ["最多 6 條"], "watch_items": [], "risks": []}
}
```

規則：只能用備忘錄正文的資訊，數字照抄正文、不推估；正文沒提到的欄位給空陣列。「營收成長但毛利率下滑」這類情況用「混合」並在 reasoning 說明。不給買賣建議。

## 3. 品質檢查與自我修正（必做）

`validate_earnings_memo`：errors 不為 0 就修正（最多 3 輪）；warnings 提到「數字在正文中找不到」時回到正文核對，抄錯就改正。

## 4. 套用並發布

```
python earnings_memo.py apply output/claude_jobs/earnings/memo_notes.json
```

輸出 `APPLIED=<場數>`、`MISSING_URLS=[]` 就完成（apply 會再跑一次品質檢查，exit 6 表示有 errors）。

## 5. 回報

一行：分析了幾場、利多／利空／混合各幾場、品質檢查結果、是否發布成功。
