AI 台股研究中心 v16 技術修正版
================================

本版以使用者上傳的 stock_dashboard(20261005-134307).py 與 config.toml 為 Dashboard 基準，
不重新改網站視覺，只補上以下技術：
1. 晨報 08:30：先更新財經資訊，再產生晨報並寄送「本次剛建立」的報告。
2. 盤後 14:30：使用最新 run_after_close_research.py，寄送盤後分析。
3. 財經資訊：08:30 與 18:00 產生 finance_info_latest.json。
4. 法說會／新聞固定以最近 2 天的實際事件日期過濾。
5. 訂閱系統：保存本機 subscribers.json；若設定 Apps Script URL，雲端與 Windows 讀取同一份訂閱名單。
6. 訂閱登記可通知管理者；晨報／盤後 Email 將有效訂閱者放入 Bcc。
7. watchlist 漲跌幅在有現價與昨收時重新計算，避免 API 單位錯誤造成 >100% 顯示。

排程：
08:30 morning（內含 finance update）
14:30 afterclose
18:00 finance
23:00 cnyes

請將下列檔案放在專案根目錄：
stock_dashboard.py
subscriber_service.py
email_agent.py
morning_report_agent.py
run_morning_report.py
run_after_close_research.py
run_finance_info_update.py
scheduled_job_runner.py
setup_research_task_v10.ps1

Cloud 訂閱同步：
google_apps_script_subscriber.gs 提供 Google Apps Script 範例。
完成 Web App 部署後，把 URL / Token 設到 Streamlit secrets 與 Windows .env：
SUBSCRIBER_SYNC_URL=...
SUBSCRIBER_SYNC_TOKEN=...

Google Apps Script 的 OWNER_EMAIL 與 SYNC_TOKEN 必須自行設定。
