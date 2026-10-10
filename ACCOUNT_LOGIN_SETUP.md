# 帳號登入與申請設定

## 功能與權限

- 未登入：可瀏覽首頁、操作策略（行情與分析資料）、財經資訊、產業分析及訂閱／聯絡頁。
- 已登入：另外可新增／刪除自選股，並使用「未來分析」。
- 帳號申請只收電子郵件，通知寄到 `a1113359@mail.nuk.edu.tw`；不會自動建立帳號或寄出密碼。
- 帳號清單只放在 Streamlit Secrets／環境變數 `AUTH_USERS_JSON`，不應提交到 GitHub。
- 密碼以 PBKDF2-HMAC-SHA256 雜湊儲存，不保存明文密碼。

## 1. 建立管理員密碼雜湊

在專案根目錄執行：

```powershell
python auth_service.py hash-password a1113359@mail.nuk.edu.tw
```

依提示輸入管理員目前要使用的密碼並確認。程式會輸出一個 JSON 物件；將輸出的雜湊值複製到下一步。不要把原始密碼放進程式碼、GitHub、一般 JSON 檔或聊天紀錄。

## 2. 本機 Streamlit Secrets

建立檔案 `.streamlit/secrets.toml`（若資料夾不存在先建立），加入：

```toml
AUTH_USERS_JSON = '{"a1113359@mail.nuk.edu.tw":"將這裡替換成 hash-password 輸出的完整雜湊字串"}'
```

將範例中的整段佔位文字替換成實際雜湊值。請確認 `.streamlit/secrets.toml` 已被 Git 忽略，不可提交。

本機使用 `streamlit run stock_dashboard.py` 啟動後，即可用管理員電子郵件與設定的密碼登入。

## 3. Streamlit Community Cloud

開啟此 App 的 **Settings → Secrets**，加入相同格式的 `AUTH_USERS_JSON`。Cloud 不會讀取你 Windows 本機的 `.streamlit/secrets.toml`，所以這一步必須另外設定。

網站寄送帳號申請通知時，也需要既有 Email 設定有效：`EMAIL_SENDER`、`EMAIL_APP_PASSWORD`、`EMAIL_RECIPIENT`。申請通知收件者固定為 `a1113359@mail.nuk.edu.tw`。

## 4. 手動建立新帳號

收到申請後，在本機執行：

```powershell
python auth_service.py hash-password 使用者電子郵件
```

將輸出 JSON 中的帳號／雜湊新增到 `AUTH_USERS_JSON` 的 JSON 物件內，保留既有帳號。例如物件的結構是：

```json
{
  "a1113359@mail.nuk.edu.tw": "pbkdf2_sha256$310000$管理員的salt$管理員的digest",
  "newuser@example.com": "pbkdf2_sha256$310000$新使用者的salt$新使用者的digest"
}
```

以上只是格式示意，請使用程式實際產生的雜湊字串，不要照抄示例值。更新本機 Secrets 與 Cloud Secrets 後，新帳號才可登入。不要把密碼或雜湊提交到公開 GitHub。

## 注意事項

目前專案既有自選股清單仍是共用清單，登入限制的是新增／刪除權限，並未改成每個帳號各自保存一份自選股。若未來需要每位使用者獨立清單，需另加持久化資料庫。
