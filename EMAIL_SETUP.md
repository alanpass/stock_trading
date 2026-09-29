# AI 盤後財報 Email Agent v55 設定

## 1. 建立 .env

複製：

```text
.env.example
↓
.env
```

然後設定：

```env
EMAIL_ENABLED=true
EMAIL_AUTO_SEND=true
EMAIL_SENDER=你的Gmail@gmail.com
EMAIL_APP_PASSWORD=你的16位AppPassword
EMAIL_RECIPIENT=a1113359@mail.nuk.edu.tw
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_SSL=false
EMAIL_ATTACH_REPORT=true
```

## 2. Gmail 驗證

需要讓寄件 Gmail 帳戶可以使用 App Password。App Password 不等於一般 Gmail 密碼，也不要貼到聊天室。

Google Gmail 的 SMTP 設定可使用：

- SMTP server：`smtp.gmail.com`
- 587：TLS / STARTTLS
- 465：SSL/TLS
- 需要驗證

## 3. 測試 Email Agent

在專案目錄：

```powershell
python email_agent.py --status
```

確認顯示：

```text
configured: true
```

再測試最新盤後報告：

```powershell
python email_agent.py --send-latest
```

若要忽略當日已寄送標記重新寄：

```powershell
python email_agent.py --send-latest --force
```

## 4. 自動寄送

Windows Task Scheduler 原本的：

```text
TaiwanStock_ModelResearch_Agent
```

不需要另外建立第二個任務。

每個工作日 15:30：

```text
盤後研究
  ↓
產生 research_YYYY-MM-DD.json / .md
  ↓
Email Agent
  ↓
寄送 HTML 盤後財報
  ↓
data/email/sent_YYYY-MM-DD.json
```

同一天若排程重跑，預設不重複寄信。

## 5. Dashboard

Dashboard 的「模型研究 Agent」裡會出現：

```text
AI 盤後財報 Email Agent
```

可以查看設定狀態，也可以手動按：

```text
寄送最新盤後財報
```

## 6. 安全注意事項

- 不要把 Gmail 一般登入密碼寫入程式。
- 不要把 App Password 寫入 ZIP、Git、公開貼文或聊天室。
- 正式使用前請確認 `.env` 沒有被同步到公開儲存庫。
