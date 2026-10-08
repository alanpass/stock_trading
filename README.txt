AI 台股研究中心｜Email / 財經資訊 / 排程 v16

本包刻意不包含 stock_dashboard.py。
本版只更新：
- email_agent.py
- morning_report_agent.py
- run_morning_report.py
- run_after_close_research.py
- run_finance_info_update.py
- scheduled_job_runner.py
- subscriber_service.py
- setup_research_task_v10.ps1（排程設定原檔）

請將以上檔案放回原專案根目錄：
C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard

手動測試：
python .\scheduled_job_runner.py morning
python .\scheduled_job_runner.py afterclose
python .\scheduled_job_runner.py finance

正確指令不要寫成：
python run scheduled_job_runner.py morning

排程：
08:30 Morning + Finance info
14:30 After-close + Email
18:00 Finance info
23:00 CNYES nightly crawl

Email：
- 晨報新聞與法說會只取前一日 00:00 至目前
- 每則新聞壓成 1~3 點
- 關鍵詞以紅字突出
- 法說會壓成 1~3 點
- 發送時自動加入目前有效訂閱者
- 訂閱者由 subscriber_service.py 保存，可選擇遠端同步
