---
name: trading-review
description: 紙上交易（零股當沖引擎、題材波段組合）的績效檢討。收盤後寫當沖檢討報告、使用者問「今天結果如何」「累計賺賠多少」「勝率多少」、或要匯出交易紀錄 Excel 時使用。不用於盤中盯盤或個股研究。
---

# 紙上交易檢討

數字一律由 `scripts/perf_summary.py` 從 `output/daytrade/*/summary.json`、`output/swing/` 計算，不要自己加總或估算。

## 步驟

1. 彙整數字：
   ```
   python .claude/skills/trading-review/scripts/perf_summary.py --xlsx
   ```
   產生當沖逐日／分策略、波段淨值與持股表，並寫 `output/trading_performance.xlsx`（使用者要 Excel 時把這個檔案給他）。
2. 當天細節：讀 `output/daytrade/<日期>/events.json`、`claude_watch.md`；用 tw-stock-research MCP 的 `get_quote`、`get_price_history` 查成交股與被擋下訊號的實際走勢（開高低收、最高點時間）。
3. 寫 `output/daytrade/<日期>/report.md`，結構與欄位見 [reference.md](reference.md)。
4. 回報給使用者（繁體中文）：今天損益、累計損益與勝率、最重要的一條檢討。

## 原則

- **先講結果，再講原因**；虧損就直接說虧損，不用「在預期範圍內」帶過。
- 連續虧損或策略累計為負時，明確建議「停止下單、只記錄」，並引用 `strategy-gate` 的結果（`output/strategy_gate/*.json`）。
- 不改程式、不碰 XQ 或券商、不給投資建議。
