V72：最近 5 日法說會 Agent 真正修復

這版只替換：
- earnings_call_agent.py
- stock_dashboard.py
- test_earnings_agent.py

不碰：output/、data/、models/、.env。

V72 修正：
1. 不再只讀最後一個 fugle_earnings_memo/probe JSON；會合併最近 5 日全部資料。
2. probe 不會覆蓋正式 Agent 結果。
3. Fugle 主題頁若延遲更新，最近 5 日會用日期搜尋補齊文章 URL。
4. Dashboard 若沒有最新的 Agent 法說包，session 首次進入會自動執行一次 5 日 Earnings Agent。
5. 每篇先由 Qwen3 呼叫 read_fugle_memo，再輸出一句話摘要、財務、營運、展望、利多、利空、風險、Q&A。
6. 原文只放在「已驗證正文」區塊，摘要成為主要內容。
7. UI 最多顯示 50 篇近期法說卡片，不再只顯示一篇。
8. +1/+5 交易日反應仍由 Fugle 歷史日K計算，不足未來交易日就留空。

套用：
cd C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard_patch_v72
Set-ExecutionPolicy -Scope Process Bypass
.\APPLY_V72.ps1

測試 Agent：
cd C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard
python test_earnings_agent.py --days 5 --limit 100

只測正文：
python test_earnings_agent.py 6582 --days 5 --crawl-only

啟動：
python -m streamlit run stock_dashboard.py
