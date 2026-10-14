# 趨勢型態功能設定

## 功能內容

- 分析 industry_rotation.py 的 ROTATION_THEMES 全部概念股，不只自選股。
- 分析及型態辨識採用 Fugle `adjusted=True` 還原日 K，降低股票分割、股票股利與除權息造成的價格基準斷層；還原後若仍發現 35% 以上單日跳空，股票會先進入「資料調整檢查」，暫停方向判斷。
- 以最近 40 個交易日的還原日 K 辨識頭肩頂、雙頂／雙底、圓底、杯柄、上升／下降楔形、旗形與三角形；另用約 6 個月的高低位與報酬判斷「由高點回落／由低點回升」的背景。
- 類別分為「資料調整檢查」、「底部反轉候選」、「多頭延續／持續創高」、「整理待突破」、「高檔轉弱／反轉警示」、「下降趨勢／弱勢整理」及「中性觀察」。
- 追蹤箱型／三角形壓力突破、連跌後 3 日轉穩或多頭上漲後 3 日轉弱。沒有重大變化時不更新公開快照、不建立 GitHub commit。
- 每檔個股展開後，可按按鈕抓取 Fugle 最新近 6 個月還原日 K，查看真實 OHLC K 棒、成交量，並可切換 MA5、MA10、MA20、MA60、MA120、MA240；排除 OHLC 缺值列，並以交易日類別軸移除週末／休市日空檔。

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

## GitHub 與網頁同步機制

- 本機程式先讀取 GitHub \`main\` 上目前已發布的 \`output/research_reports/trend_patterns_public.json\` 作為比較基準，不依賴本機是否為 Git clone；因此沒有 \`.git\` 也能發布。
- 初次建立快照時，所有成功分析的概念股會寫入該 JSON。之後只在分類／型態狀態重大改變時，使用 GitHub Contents API 更新同一個 \`main\` 分支上的 JSON。
- 網站程式讀取同一路徑的 JSON。若 Streamlit Community Cloud 綁定此 GitHub repository 的 \`main\` 分支並允許自動部署，JSON commit 會觸發網站重新部署，部署完成後網頁便讀到新結果。
- 使用者層級的 \`GITHUB_TOKEN\` 或 \`GH_TOKEN\` 需具有 \`alanpass/stock_trading\` 的 Contents 寫入權限。Windows 工作排程器若未繼承環境變數，程式也會嘗試從目前使用者的環境變數登錄值讀取；Token 不會寫入 log。
- 若本次沒有重大變化，會保留已發布快照、不建立 GitHub commit，這是刻意設計，並非同步失敗。

第一次同步請在本機專案資料夾執行：

    python .\run_trend_pattern_update.py --force

請在輸出中確認 \`github.ok = true\` 並取得 \`commit_sha\`；若執行結果表示沒有重大變化，請查看 GitHub 上的公開 JSON 是否已經有股票資料。若仍是初始空快照，請檢查 \`FUGLE_API_KEY\`、\`GITHUB_TOKEN\` 及 \`output\\scheduler_logs\\trend_patterns_runner.log\`。

網站上的個股近六個月 K 棒圖則在使用者按下按鈕後才透過 Fugle 查詢，避免一次對所有股票發送 API 請求。

## 需要留意的前提

- 每日 19:00 的排程依賴本機 Windows 可執行；電腦關機時不會準時跑。
- 要取得最新個股圖，Streamlit Cloud Secrets／本機環境須有可用的 FUGLE_API_KEY。
- 初次執行需有足夠的歷史日 K 快取或 Fugle API 存取權限。
