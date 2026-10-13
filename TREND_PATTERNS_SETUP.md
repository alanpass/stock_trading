# 趨勢型態功能設定

## 功能內容

- 分析 industry_rotation.py 的 ROTATION_THEMES 全部概念股，不只自選股。
- 以近 3 個月日 K 辨識頭肩頂、雙頂／雙底、圓底、杯柄、上升／下降楔形、旗形與三角形；再用約 6 個月的高低位與報酬判斷「由高點回落／由低點回升」的背景。
- 類別分為「底部反轉候選」、「多頭延續／持續創高」、「整理待突破」、「高檔轉弱／反轉警示」、「下降趨勢／弱勢整理」及「中性觀察」。
- 追蹤箱型／三角形壓力突破、連跌後 3 日轉穩或多頭上漲後 3 日轉弱。沒有重大變化時不更新公開快照、不建立 GitHub commit。
- 每檔個股展開後，可按按鈕抓取 Fugle 最新近 3 個月日 K，查看收盤價、MA20、MA50 和成交量。

型態辨識為可解釋的規則式估算，不保證後續走勢。IG 的說明包含頭肩、雙頂、雙底、圓底、杯柄、楔形、旗形及三角形；本功能以程式規則近似這些型態。

## 安裝每日 19:00 排程（Windows）

先將本分支合併至 main 並同步至本機專案資料夾。確認目前資料夾包含 run_trend_pattern_update.py 後，在 PowerShell 執行：

    cd C:\Users\10501\OneDrive\Desktop\stock_trading\stock_dashboard
    powershell -ExecutionPolicy Bypass -File .\setup_trend_pattern_task.ps1

第一次先手動測試（會建置／更新本機 JSON，並嘗試發布重大變化）：

    python .\run_trend_pattern_update.py --force

只想測試計算與輸出、不寫入 GitHub：

    python .\run_trend_pattern_update.py --force --no-publish

排程記錄：
output\scheduler_logs\trend_patterns_runner.log 與 output\scheduler_logs\trend_patterns_YYYY-MM-DD.log

## GitHub 發布條件

使用 GITHUB_TOKEN 或 GH_TOKEN，需要 alanpass/stock_trading 的 Contents 讀寫權限。若初次建立快照，所有成功辨識的股票都會列入；日後只在分類／型態狀態重大改變時改寫 output/research_reports/trend_patterns_public.json。網站上的個股三個月圖則在使用者按下按鈕後才透過 Fugle 查詢，避免一次對所有股票發送 API 請求。

## 需要留意的前提

- 每日 19:00 的排程依賴本機 Windows 可執行；電腦關機時不會準時跑。
- 要取得最新個股圖，Streamlit Cloud Secrets／本機環境須有可用的 FUGLE_API_KEY。
- 初次執行需有足夠的歷史日 K 快取或 Fugle API 存取權限。
