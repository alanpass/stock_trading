# Claude 研究 Agent 架構

本專案所有需要 LLM 的研究工作，都由 Claude 在這台 Windows 電腦上執行；專案內沒有 Ollama / Qwen3 程式碼。
Python 只負責「抓資料、計算、寄信、發布」，Claude 負責「閱讀、判斷、查證、撰寫」。

## 一、組成

| 組成 | 是什麼 | 在專案裡的位置 |
|---|---|---|
| **MCP server** `tw-stock-research` | 22 個結構化工具：即時市場資料、研究包查證、流程維運、品質檢查 | `mcp_server/tw_stock_server.py`，註冊在 `.mcp.json`（`.claude/settings.json` 啟用） |
| **Agent Skills** | 每個流程的操作手冊：步驟、該用的工具、輸出 JSON 格式、規則、自我檢查 | `.claude/skills/<流程>/SKILL.md` |
| **品質把關** | 檢查 Claude 寫的結果：漏寫自選股、漲跌幅與行情不符、連結不在研究包、投資建議用語… | `quality_checks.py`；MCP `validate_*` 與 finalize／apply 共用 |
| **排程任務** | Claude 桌面版 Scheduled tasks，到時間開一個 Claude session 執行對應 Skill | 使用者資料夾 `.claude/scheduled-tasks/tw-stock-*/SKILL.md` |
| **命令列工具** | MCP 不可用時的備援，功能相同 | `claude_tools.py`、`pipeline_status.py`、`finance_digest.py`、`earnings_memo.py`、`stock_notes.py` |

### MCP 工具一覽

| 組別 | 工具 |
|---|---|
| 市場資料 | `get_quote`、`get_price_history`、`get_institutional_flow`、`get_holder_structure`、`get_market_status` |
| 研究包查證 | `research_market_overview`、`research_company_financial`、`research_company_profile`、`research_earnings_events`、`research_read_fugle_memo`、`research_news_batch`、`research_search_news`、`research_search_live_news`、`research_theme_evidence`、`research_previous_notes` |
| 流程維運 | `pipeline_status`、`get_job_brief`、`get_stock_note` |
| 品質把關 | `validate_afterclose_notes`、`validate_morning_notes`、`validate_finance_digest`、`validate_earnings_memo` |

MCP server 只提供讀取與檢查；寄信、發布仍由 Skill 執行 `run_*.py --finalize` 或 `apply`，這些入口會**再跑一次品質檢查**，有 errors 就拒絕寄出（exit 6）。
測試：`python test_mcp_server.py <日期>`（以真正的 stdio 連線呼叫 server）。

### Agentic 設計

- **自主查證**：Skill 只規定要回答的問題與輸出格式，查哪些資料、查幾次由 Agent 依證據決定。
- **自我檢查迴圈**：寫完 → `validate_*` → 依 errors 修正 → 再檢查，最多 3 輪，仍有錯誤就不寄出。
- **多 Agent 分工**：盤後研究可把 20 多檔自選股分組交給 subagent 並行查證，再由主 Agent 合併與檢查。
- **跨日記憶**：每天先讀上一期研究筆記，寫出哪些假設被證實或推翻。
- **維運 Agent**：`pipeline-health` 每天兩次檢查全部流程，自動重跑安全的步驟（新聞更新、網站發布、研究包），會寄信的步驟只回報、不自動重寄。

## 二、每日流程

```
Windows 工作排程器（資料）                Claude 排程任務（研究）
─────────────────────────────           ─────────────────────────────────────
08:10 11:00 13:30 16:00 23:00            11:40 16:40 23:40  finance-news-digest
  財經新聞爬蟲＋規則摘要  ───────────▶     AI 今日重點＋逐篇新聞重點 → 發布網站
08:30 早報研究包         ───────────▶     08:40  morning-report → 寄早報
                                           10:30／20:30  pipeline-health → 檢查、修復、回報
14:30 盤後研究包（官方行情未更新會重試） ─▶ 15:45  after-close-research
                                           → 盤後研究＋自選股／熱門股個股筆記
                                           → 寄盤後信、個股筆記發布網站
18:00 Fugle 法說會正文＋規則重點 ─────▶   18:40  earnings-call-memo → 法說會摘要 → 發布
隨時（使用者要求）                         stock-deep-research → 單檔深度研究 → 發布
```

（排程任務實際執行時間會晚幾分鐘，這是系統的負載平衡延遲。）

每個 Claude 流程都是同一個模式：

1. **prepare／export**：Python 把當天資料整理成研究包（`output/claude_jobs/<流程>_<日期>/package.json`）與精簡摘要 `brief.md`。
2. **研究**：Claude 依 Skill 讀 brief，用 MCP 工具（或 `claude_tools.py`）深入查證，寫出結果 JSON。
3. **品質檢查**：`validate_*` 找出錯誤，Claude 修正後再檢查。
4. **finalize／apply**：Python 再驗證一次，寫回報告、寄信、發布到 GitHub `finance-data` 分支。
   盤後研究與早報都有 `--dry-run`，只產生 Email 預覽、不寄信不發布。

## 三、檔案對照

| 流程 | Skill | Python 入口 | Claude 輸出 |
|---|---|---|---|
| 盤後研究 | `after-close-research` | `run_after_close_research.py --prepare / --finalize` | `agent_notes.json`（含 `stock_notes`） |
| 早報 | `morning-report` | `run_morning_report.py --prepare / --finalize` | `morning_notes.json` |
| 財經 AI 重點 | `finance-news-digest` | `finance_digest.py export / apply` | `digest_notes.json` |
| 法說會 | `earnings-call-memo` | `earnings_memo.py export / apply` | `memo_notes.json` |
| 流程維運 | `pipeline-health` | `pipeline_status.py` | 回報（自動修復安全步驟） |
| 個股深度研究（隨時） | `stock-deep-research` | `stock_notes.py upsert` | `output/claude_jobs/deep/<代號>_<日期>.json` |

共用設定在 `ai_backend.py`（狀態標籤：`claude_agent`＝完成、`pending_claude`＝等待 Claude）。

## 四、網站如何顯示

網站（含 Streamlit Cloud）**不會即時呼叫任何 LLM**，只讀取已發布的結果並標示產生時間：

- 未來分析 → 「04｜AI 個股研究筆記」（自選股／當日熱門股／深度研究）：`stock_notes_public.json`（`stock_notes_ui.py`）
- 操作策略 → 進場建議區塊、市場情報的盤後歸因：顯示該股票的 Claude 個股筆記
- 財經資訊 → 「AI 今日重點」、每則新聞重點、法說會摘要：`finance_info_public.json`
- 未來分析 → 模型研究 Agent：讀最新 `research_<日期>.json`

## 五、金鑰

Fugle 金鑰只放在 `.env`（本機）與 Streamlit Cloud 的 App Settings → Secrets（雲端），程式碼不再寫死。
`stock_api.py` 依序讀取：環境變數 → `.env` → `.streamlit/secrets.toml`。

## 六、手動執行與維護

- 手動補跑任一流程：在 Claude 對話中說「執行 after-close-research skill」，或在側邊欄 Scheduled 裡對該任務按 **Run now**。
- 研究單一股票：對 Claude 說「研究一下 2330」（`stock-deep-research`）。
- 看今天流程是否正常：「今天流程正常嗎」（`pipeline-health`），或 `python pipeline_status.py`。
- 排程任務只在電腦開機、Claude 桌面版開著時執行；錯過的時段會在下次開啟時補跑。
- 第一次請對每個任務按一次 Run now 並允許需要的指令與 MCP 工具，之後就不會再停下來詢問。
- Email 寄送失敗時 Skill 會回報原因，不會重複寄信。
- 要新增 AI 流程：照同樣模式寫一個 prepare／apply 腳本、在 `quality_checks.py` 加檢查、寫 `.claude/skills/<名稱>/SKILL.md`，再建立排程任務。
