# 產業分析更新套件

本套件為 stock_trading 新增「產業分析」主題，Windows 在每個台股交易日 15:10 更新資料，再將精簡快照發布到 GitHub。

## 功能

- 27 個 AI 供應鏈子產業主題，預設一次顯示全部群組；每個主題都有寫入程式的代表成分股代號與名稱，頁面會展開完整分類名單。同一檔股票可以出現在多個主題。
- 優先使用 TWSE 官方加權指數歷史資料當基準；個股歷史日 K 由 Fugle 取得。
- 顯示今日／昨日、5／20／60 日、今年以來平均報酬、上漲／下跌家數、52 週新高、RRG 四象限、成分股前十強弱。
- 新增「產業族群平均漲跌幅」水平長條圖，預設顯示最新已完成交易日，可切換前一交易日；紅色代表族群平均上漲、綠色代表平均下跌。
- 族群平均漲跌幅＝該主題有有效行情的代表股日漲跌幅算術平均，每檔代表股等權計算；此為研究用代表股指標，不等同官方產業指數。
- 新增「美股產業指標股資訊」節點，將 27 個台股供應鏈子產業對照到美股代表公司，列出代號、公司名稱、產業對照角色、最新可取得價格、當日漲跌額／漲跌幅與美東資料日期；行情採 Yahoo Finance chart endpoint，15 分鐘快取，無需付費 API Key。
- 排行表預設依最新交易日族群平均漲跌幅排序，漲跌相關數值依台股慣例紅漲綠跌。
- 僅在取得可驗證的當日加權指數資料後發布。若資料未更新、品質檢查失敗或遇休市日，保留上一份成功快照。

## 安裝更新

1. 把套件檔案解壓縮到現有 stock_dashboard 專案根目錄。
2. 用 PowerShell 執行 setup_industry_rotation_task.ps1，或使用最新本地更新套件。
3. 確認 Windows 使用者環境變數 GITHUB_TOKEN 已設定，且 Token 具備 alanpass/stock_trading 的 Contents 寫入權限。
4. 手動測試可執行 python run_industry_rotation_update.py --force；此指令會請求行情並嘗試發布，請確認 Token 與 API 設定正常。
5. 查看 logs/industry_rotation_task.log 確認資料日期、股票樣本數與 GitHub commit。

## 每日更新架構

Windows Task Scheduler（週一至週五 15:10）
→ 檢查台股交易日與 TWSE 官方加權指數日期
→ 增量更新代表股歷史日 K
→ 計算產業等權指數、相對強弱、動能、績效與族群平均漲跌幅（%）
→ 查詢美股供應鏈指標股行情並於網站端快取 15 分鐘
→ 寫入 output/research_reports/industry_rotation_public.json
→ 使用 GITHUB_TOKEN 更新同一路徑到 main
→ Streamlit Cloud 重新部署後顯示最新資料

排程會自動略過週末與官方休市日。電腦需開機、可連網，而且建立排程的 Windows 使用者需登入；Streamlit Cloud 不負責執行 15:10 的資料蒐集。

## 專案檔案

- 新增程式：industry_rotation.py、run_industry_rotation_update.py、us_industry_indicators.py
- 修改程式：stock_dashboard.py
- 新增設定：設定產業分析排程.ps1
- 公開快照：output/research_reports/industry_rotation_public.json
- 行情快取建立在 data/rotation/；排程記錄在 logs/industry_rotation_task.log。

本套件不會把本機 .env、API Token 或模型檔案寫入公開 JSON。
