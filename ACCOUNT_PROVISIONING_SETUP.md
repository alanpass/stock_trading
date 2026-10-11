# 帳號自動建置系統設定

本功能使用 Supabase Postgres 作為雲端永久資料庫。Streamlit Cloud 本機磁碟不作為資料庫。

## 1. 建立資料庫
1. 登入 Supabase，建立一個專案。
2. 開啟 SQL Editor，執行 account_provisioning_schema.sql 全部內容。
3. 到 Project Settings → API，取得 Project URL 與 service_role key。service_role key 是高權限密鑰，不可放入前端、程式碼或 GitHub。

## 2. 設定 Streamlit Community Cloud Secrets
在網站 App → Settings → Secrets 加入下列項目；保留原有的 AUTH_USERS_JSON、AUTH_SESSION_SECRET 與 Email SMTP 設定：

~~~toml
ACCOUNT_DB_URL = "https://你的專案代碼.supabase.co"
ACCOUNT_DB_SERVICE_ROLE_KEY = "你的 service_role key"
ACCOUNT_PUBLIC_BASE_URL = "https://stocktrading-c66hschtsd2qh6vkyv4mtq.streamlit.app/"
~~~

ACCOUNT_PUBLIC_BASE_URL 必須是實際部署網站的公開網址。

## 3. 審核流程
1. 使用者在網站申請帳號。
2. 系統將申請與兩組一次性 token 的 SHA-256 雜湊寫入 Supabase，寄送管理員審核郵件。
3. 管理員點擊同意／拒絕連結，網站會先顯示申請資料與確認按鈕。只有再次按下確認才會改變狀態，避免郵件安全掃描器預先開啟連結時誤觸審核。
4. 核准時，系統產生 8 位英數初始密碼，將 PBKDF2-HMAC-SHA256 密碼雜湊自動保存到 Supabase；明文密碼只寄給申請者。
5. 登入時，既有帳號仍從 AUTH_USERS_JSON 驗證；由申請流程建立的新帳號，則由伺服器自動讀取 Supabase 中最新一筆已核准且已到啟用時間的密碼雜湊，不需要管理員手動更新 AUTH_USERS_JSON。
6. 核准時間起 5 分鐘後，資料庫才允許該帳號登入；5 分鐘內無法登入。
7. 拒絕時，系統寄送拒絕通知給申請者，該筆申請不會取得可登入的憑證。

## 4. 審核連結安全性
- 審核 token 為隨機值，資料庫只保存 SHA-256 雜湊。
- 連結有效期限 48 小時；審核成功後只能執行一次。有效期限內重新開啟舊連結會顯示已結案狀態，不會再次核准或拒絕。
- 開啟連結不會直接核准／拒絕，必須再按一次網站確認按鈕，防止郵件掃描器自動點開 URL。
- 不要把 Supabase service_role key 加入 GitHub。

## 5. 部署前檢查
- ACCOUNT_DB_URL、ACCOUNT_DB_SERVICE_ROLE_KEY、ACCOUNT_PUBLIC_BASE_URL 已設定。
- Supabase SQL 已執行且 account_requests 表存在。
- Streamlit Secrets 的 SMTP 設定仍有效。
