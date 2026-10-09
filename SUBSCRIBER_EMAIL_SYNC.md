# 訂閱者每日早報／盤後分析寄送設定

本專案的早報與盤後分析由 Windows 本機排程產生，之後由 `EmailAgent` 寄送。網站則部署在 Streamlit Community Cloud。兩端必須共用同一份有效訂閱名單，否則雲端表單收集到的地址不會自動出現在本機寄信程序中。

## 1. 建立 Google Apps Script 共享名單

專案已有 `google_apps_script_subscriber.gs` 範本。

1. 開啟 [Google Apps Script](https://script.google.com/) 並建立新專案。
2. 將範本檔的完整內容貼入 Apps Script 編輯器。
3. 在 Apps Script 編輯器中，設定 `OWNER_EMAIL` 為管理者信箱，並把 `SYNC_TOKEN` 改成自行產生的長隨機字串。不要把實際 Token 寫回 GitHub。
4. 選擇 **Deploy → New deployment → Web app**。
5. **Execute as** 選擇 **Me**；**Who has access** 設為 **Anyone**，讓 Streamlit Cloud 和 Windows 排程可以呼叫。雖然 Web App 可被呼叫，名單 API 仍會檢查 `SYNC_TOKEN`；請使用高熵、不可猜測的 Token。
6. 授權並部署，複製產生的 Web app URL（通常以 `/exec` 結尾）。

第一次成功呼叫時，範本會自動建立 Google 試算表與 `Subscribers` 工作表。請勿把試算表 ID 或 Token 寫進公開程式碼。

## 2. 設定 Streamlit Community Cloud

在 Streamlit Cloud 開啟網站的 **Settings → Secrets**，保留目前已設定的 Email 欄位，另外加入以下兩個 TOML 欄位：

```toml
SUBSCRIBER_SYNC_URL = "https://script.google.com/macros/s/你的部署ID/exec"
SUBSCRIBER_SYNC_TOKEN = "與 Apps Script 中完全相同的長隨機字串"
```

將 URL 和 Token 換成自己的設定，儲存後等待 App 重新啟動。請勿把真正的 Token 交給 GitHub 或公開貼文。

## 3. 設定 Windows 本機排程

在專案根目錄的本機 `.env` 加入同一組設定：

```env
SUBSCRIBER_SYNC_URL=https://script.google.com/macros/s/你的部署ID/exec
SUBSCRIBER_SYNC_TOKEN=與 Apps Script 中完全相同的長隨機字串
```

請沿用既有的 `EMAIL_SENDER`、`EMAIL_APP_PASSWORD` 等本機 SMTP 設定，不需要再建立一組寄件帳號。新版 `subscriber_service.py` 會先讀 Windows 環境變數，再回退讀專案 `.env`，因此 Windows Task Scheduler 不必依賴互動式 PowerShell 的臨時環境變數。

**不要把 `.env` 提交到 GitHub。** 若目前的本機專案尚未包含新版 `subscriber_service.py`，先合併此修正並更新本機專案，再進行下面的測試。

## 4. 驗證名單同步（不寄出報告）

在 PowerShell 切換到專案根目錄後執行：

```powershell
python -c "from pathlib import Path; from subscriber_service import _remote_url, _remote_token, report_recipient_emails; p=Path('.').resolve(); print('共享 URL 已設定：', bool(_remote_url(p))); print('共享 Token 已設定：', bool(_remote_token(p))); print('有效訂閱收件者數：', len(report_recipient_emails(p)))"
```

預期為：共享 URL 與 Token 都是 `True`，有效訂閱者數量大於 0。這個指令只檢查設定及讀取收件者數量，不會寄送電子郵件，也不會印出 Token 或任何訂閱者信箱。

若讀取失敗，請先確認 Apps Script 已部署成 Web App，URL 是最新的 `/exec` 網址，且 Token 與 Apps Script 編輯器中的值完全一致。

## 5. 同步既有訂閱者

啟用共享名單之前已經在 Streamlit Cloud 訂閱的人，可能只存在 Cloud 的暫存檔，不會自動回填到 Google 試算表。

共享名單設定完成後，請讓既有訂閱者在網站表單再次送出同一個姓名／Email 並勾選訂閱；系統會將資料 upsert 到共享試算表。已經是有效訂閱者的人不會因這次補同步而重複收到訂閱成功信。也可以由管理者將既有有效訂閱資料整理後匯入 `Subscribers` 工作表，確認 `subscribed` 與 `active` 欄均為 TRUE。

## 6. 寄送流程與限制

目前排程時程為台灣時間：

- 08:30：產生並寄送每日早報。
- 14:30：產生並寄送盤後分析。

寄送程序會先讀取共享名單，再將郵件個別寄給管理者與有效訂閱者。個別寄送避免訂閱者互相看見彼此的地址。

兩個工作仍然由 Windows Task Scheduler 執行，因此電腦需要在排程時間可執行（睡眠時可嘗試喚醒；完全關機時不會執行）。如果研究流程因當日行情資料驗證失敗而提前停止，該次報告不會寄送；這與訂閱名單同步是不同的問題。排程紀錄位於 `output/scheduler_logs`。
