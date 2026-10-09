# 產業分析更新套件

本套件為 stock_trading 新增「產業分析」主題，Windows 在每個台股交易日 15:10 更新資料，再將精簡快照發布到 GitHub。

## 功能

- 27 個供應鏈／子產業主題：晶圓代工、ASIC／矽智財、BMC、高速傳輸 IC、先進封裝設備、封裝測試、測試介面、記憶體、ABF、CCL、PCB、散熱、電源、重電、滑軌、機殼、連接器、被動元件、ODM、網通、矽光子、化合物半導體、低軌衛星、機器人、廠務工程、金融等。
- 優先使用 TWSE 官方加權指數歷史資料當基準；個股歷史日 K 由 Fugle 取得。
- 顯示今日／昨日、5／20／60 日、今年以來平均報酬、上漲／下跌家數、52 週新高、RRG 四象限、成分股前十強弱。
- 「樣本成交值占比變化」只表示本頁追蹤代表股的成交值聚焦代理，不冒充全市場實際資金流。
- 僅在取得可驗證的當日加權指數資料後發布。若資料未更新、品質檢查失敗或遇休市日，保留上一份成功快照。

## 安裝更新

1. 把套件檔案解壓縮到現有 stock_dashboard 專案根目錄。
2. 用 PowerShell 執行 設定產業分析排程.ps1。
3. 確認 Windows 使用者環境變數 GITHUB_TOKEN 已設定，且 Token 具備 alanpass/stock_trading 的 Contents 寫入權限。
4. 手動測試可執行 python run_industry_rotation_update.py --force；此指令會請求行情並嘗試發布，請確認 Token 與 API 設定正常。
5. 查看 logs/industry_rotation_task.log 確認資料日期、股票樣本數與 GitHub commit。

## 每日更新架構

Windows Task Scheduler（週一至週五 15:10）
→ 檢查台股交易日與 TWSE 官方加權指數日期
→ 增量更新代表股歷史日 K
→ 計算產業等權指數、相對強弱、動能、績效與樣本成交值占比
→ 寫入 output/research_reports/industry_rotation_public.json
→ 使用 GITHUB_TOKEN 更新同一路徑到 main
→ Streamlit Cloud 重新部署後顯示最新資料

排程會自動略過週末與官方休市日。電腦需開機、可連網，而且建立排程的 Windows 使用者需登入；Streamlit Cloud 不負責執行 15:10 的資料蒐集。

## 專案檔案

- 新增程式：industry_rotation.py、run_industry_rotation_update.py
- 修改程式：stock_dashboard.py
- 新增設定：設定產業分析排程.ps1
- 公開快照：output/research_reports/industry_rotation_public.json
- 行情快取建立在 data/rotation/；排程記錄在 logs/industry_rotation_task.log。

本套件不會把本機 .env、API Token 或模型檔案寫入公開 JSON。
