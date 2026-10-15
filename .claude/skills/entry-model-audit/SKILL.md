---
name: entry-model-audit
description: 進場建議模型的定期稽核。對自選股重新訓練 5 分 K 回檔模型並做前推（樣本外）驗證，找出預測力消失或恢復的股票、檢查 K 線型態與趨勢判斷是否合理，必要時提出規則調整建議。由排程任務每週六 10:00 觸發，也可在使用者問「進場模型準不準」時執行。
---

# 進場模型稽核

專案根目錄：`C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard`（所有指令都在這裡執行）。
模型與規則在 `entry_model.py`（v3），網站「操作策略 → 04｜進場建議分析系統」使用同一份。

## 1. 執行稽核

MCP：`audit_entry_model()`（全部自選股，約 5 秒／檔）；沒有 MCP 時：`python entry_analysis.py audit`。
結果會存到 `output/research_reports/entry_model_audit.json`，並和上一次比較。

## 2. 判讀

- `prob_has_skill`：回檔機率的 Brier skill ≥ 0.02（比歷史回檔頻率準）。為 false 的股票，網站會自動改用歷史頻率。
- `low_has_skill`：最低價預測誤差比「歷史中位數」小 2% 以上。為 false 時網站用「現價 − 0.5 ATR」當最佳進場價。
- `lost_skill_since_last`／`gained_skill_since_last`：和上次相比的變化。
- 有用的股票比例、`median_brier_skill` 是否明顯下降（例如中位數由 > 0.04 掉到 < 0.02）。

## 3. 抽查判斷是否合理

挑 3～5 檔（包含失去預測力的股票），呼叫 `analyze_entry(symbol)`，用 `get_price_history`、`get_quote` 核對：
- 5 分 K／日 K 型態是否符合實際 K 棒（例如標示「長紅K」時，當根實體是否真的大）。
- 日線趨勢（MA20／MA60）是否和實際走勢一致。
- 動作（適合回檔進場／等待回檔確認／不追高／不建議）是否和趨勢、型態、評分一致。

## 4. 回報

不修改程式碼。用條列回報：
- 有用的回檔機率模型 N／M 檔、最低價模型 N／M 檔、Brier skill 中位數，與上次比較。
- 失去／恢復預測力的股票。
- 抽查發現的判斷問題（具體股票、日期、實際 K 棒數字），以及建議的規則調整，交給使用者決定是否修改 `entry_model.py`。
