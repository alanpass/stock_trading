# -*- coding: utf-8 -*-
"""AI 盤後財報 Email Agent

用途：

1. 讀取盤後 ResearchAgent 產生的 JSON / Markdown 報告。
2. 將市場漲跌、產業強弱、法說會、利多／利空、財報/營收、Qwen3 摘要整理成 HTML Email。
3. 使用 Gmail SMTP 寄送，不需要 OpenAI API，也不需要把 Gmail 密碼寫進程式。
4. 以本機 .env / Windows 環境變數保存 Gmail 帳號與 App Password。

預設 Gmail 設定：
smtp.gmail.com / 587 / STARTTLS
Google 官方目前仍提供 smtp.gmail.com 的 SMTP 設定；587 使用 TLS/STARTTLS，
也可使用 465 SSL。使用程式登入時應使用 OAuth 或 App Password，而不是把一般 Gmail
登入密碼放進程式。

本版（Email 排版強化）：
- 移除「今日先看這裡」與後方章節之間逐字重複的長段落（法說會摘要、模型結論只保留一處）。
- 財報/營收數字改用「億元」與百分比格式，取代原始未格式化的浮點數。
- 關鍵漲跌、法說訊號、營收年增率等重點改用粗體紅字（漲／利多／重點）與綠字（跌／利空）標示，
  並比照台股慣例（紅漲綠跌）；接近漲跌停另加標籤。
- 「最後選取標的盤後事件歸因」不再輸出原始 JSON，改為結構化重點摘要。
- 本機 Qwen3 8B 的文字只顯示結論；完整推理預設不放進信件，EMAIL_SHOW_LLM_REASONING=true 才以可展開區塊附上。
- 表格中整欄相同的內容（例如每列都相同的「無法歸因」原因、相同的證據等級）改為只在表上方說一次。
- 新增 `python email_agent.py --preview`：只產生 HTML 預覽檔，不寄信。
- 拿掉「模型健康度」整節、拿掉法說會全文區塊（不再出現在盤後報告；model_health/earnings_calls
  這兩份資料本身都還在 JSON 裡，只是盤後 Email 不顯示，其他消費者不受影響）。
- 拆成「早報」與「盤後報告」兩封信：本檔案（EmailAgent）只負責盤後報告；早報改由
  MorningEmailAgent（見 morning_email_agent.py，繼承本類別、重用同一套排版與寄信邏輯）負責，
  內容包含最近 5 天法說會（重用 `_earnings_section`）、近 2 天財經新聞摘要（見
  cnyes_news_agent.py）、AI 觀察的看好產業與個股（見 morning_report.py）。
- 盤後報告重新以「① 今日 AI 研究重點」取代原本的「今日先看這裡」：依正向／負向／待追蹤分組，
  不再依資料種類分組；法說會的風險摘句、題材的查證提醒只在這裡出現一次，下面章節不再重複整段話。
- 漲跌幅表格改成只取前 5 名、產業強弱只取前 6 名、營收成長只取前 10 名；「可能原因」欄改成
  「相關題材」，個股掛在某個利多題材下才顯示題材名稱（例如 PCB、記憶體），沒有的話留白，
  不再重複官方產業分類、也不再寫「缺乏新聞、無法歸因」這類制式文字。
- 章節依序為：一 今日市場重點、二 利多題材、三 營收成長重點、四 本機 Qwen3 8B 研究摘要、
  五 最後選取標的盤後事件歸因、六 資料來源與查證。
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from email.utils import formataddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

TAIPEI_TZ = "Asia/Taipei"
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_RECIPIENT = "a1113359@mail.nuk.edu.tw"


# ---------------------------------------------------------------------------
# .env loader：不依賴 python-dotenv，維持本專案 requirements 精簡。
# 系統環境變數優先於 .env。
# ---------------------------------------------------------------------------
def load_local_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # 支援簡單的單/雙引號包覆。
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    except Exception:
        return {}
    return values


def _cfg_value(env_values: dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key, env_values.get(key, default)).strip()


def _as_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class EmailConfig:
    enabled: bool
    auto_send: bool
    smtp_host: str
    smtp_port: int
    sender: str
    app_password: str
    recipient: str
    display_name: str
    subject_prefix: str
    attach_reports: bool
    use_ssl: bool
    show_llm_reasoning: bool = False


class Raw(str):
    """標記一段已經是安全 HTML 的字串，交給 `_esc()` 時不再重複跳脫。

    所有 `Raw(...)` 的內容都必須由本檔案自行組成（例如顏色 span、徽章），
    任何來自報告資料的文字仍須先經過 `html.escape()`，才能包進 `Raw`。
    """


class EmailAgent:
    """將盤後研究報告轉成 HTML Email 並透過 SMTP 寄出。"""

    _CONCLUSION_WORDS = ("總結", "結論")

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()
        self.env_path = self.base / ".env"
        self.env = load_local_env(self.env_path)
        self.output_dir = self.base / "output" / "research_reports"
        self.email_dir = self.base / "data" / "email"
        self.email_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> EmailConfig:
        def first(*keys: str, default: str = "") -> str:
            for key in keys:
                value = _cfg_value(self.env, key, "")
                if value:
                    return value
            return default

        try:
            port = int(first("EMAIL_SMTP_PORT", default=str(DEFAULT_SMTP_PORT)))
        except Exception:
            port = DEFAULT_SMTP_PORT

        return EmailConfig(
            enabled=_as_bool(first("EMAIL_ENABLED", default="true"), True),
            auto_send=_as_bool(first("EMAIL_AUTO_SEND", default="true"), True),
            smtp_host=first("EMAIL_SMTP_HOST", default=DEFAULT_SMTP_HOST),
            smtp_port=port,
            sender=first("EMAIL_SENDER", "EMAIL_GMAIL_ACCOUNT"),
            app_password=first("EMAIL_APP_PASSWORD", "GMAIL_APP_PASSWORD").replace(" ", ""),
            recipient=first("EMAIL_RECIPIENT", default=DEFAULT_RECIPIENT),
            display_name=first("EMAIL_DISPLAY_NAME", default="AI 台股盤後財報 Agent"),
            subject_prefix=first("EMAIL_SUBJECT_PREFIX", default="【AI 台股盤後財報】"),
            attach_reports=_as_bool(first("EMAIL_ATTACH_REPORT", default="true"), True),
            use_ssl=_as_bool(first("EMAIL_USE_SSL", default="false"), False),
            show_llm_reasoning=_as_bool(first("EMAIL_SHOW_LLM_REASONING", default="false"), False),
        )

    # ----------------------------- status ---------------------------------
    def status(self) -> dict[str, Any]:
        c = self.config
        missing = []
        if not c.sender:
            missing.append("EMAIL_SENDER")
        if not c.app_password:
            missing.append("EMAIL_APP_PASSWORD")
        if not c.recipient:
            missing.append("EMAIL_RECIPIENT")
        configured = c.enabled and not missing
        return {
            "enabled": c.enabled,
            "auto_send": c.auto_send,
            "configured": configured,
            "sender": c.sender,
            "recipient": c.recipient,
            "smtp_host": c.smtp_host,
            "smtp_port": c.smtp_port,
            "use_ssl": c.use_ssl,
            "attach_reports": c.attach_reports,
            "missing": missing,
            "env_path": str(self.env_path),
        }

    # ----------------------------- dedupe ----------------------------------
    def _marker_path(self, report_date: str) -> Path:
        return self.email_dir / f"sent_{report_date}.json"

    def already_sent(self, report_date: str) -> bool:
        return self._marker_path(report_date).exists()

    def _write_marker(self, report_date: str, payload: dict[str, Any]) -> None:
        self._marker_path(report_date).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----------------------------- HTML 基礎工具 ----------------------------
    @staticmethod
    def _esc(value: Any) -> str:
        if isinstance(value, Raw):
            return str(value)
        return html.escape(str(value if value is not None else ""))

    @staticmethod
    def _pct(value: Any, multiplier: float = 100.0) -> str:
        try:
            return f"{float(value) * multiplier:+.2f}%"
        except Exception:
            return "—"

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        # 報告 JSON 內常有 NaN（例如同業中位數算不出來）；NaN/inf 一律視為缺值，
        # 避免版面出現「+nan%」，也避免 NaN 讓排序結果失真。
        try:
            x = float(value)
        except Exception:
            return default
        return x if math.isfinite(x) else default

    def _table(self, headers: list[str], rows: list[list[Any]]) -> str:
        if not rows:
            return "<p class='muted'>目前沒有可用資料。</p>"
        head = "".join(f"<th>{self._esc(h)}</th>" for h in headers)
        body = []
        for row in rows:
            cells = "".join(f"<td>{self._esc(v)}</td>" for v in row)
            body.append(f"<tr>{cells}</tr>")
        return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"

    # ------------------------- 重點標示（紅漲綠跌） --------------------------
    # 以下工具把數字轉成「粗體紅字／綠字」等視覺重點，統一比照台股慣例：
    # 正值（上漲、利多、買超、營收成長）＝紅字，負值（下跌、利空、賣超）＝綠字。
    # 欄位命名為 *_percent / *_pct / revenue_yoy / strength 等，資料本身已是
    # 「1.0 代表 1%」的百分比數字；只有 *_return（如 post_1d_return）是小數，
    # 需要 as_fraction=True 額外乘以 100，呼叫前務必確認欄位定義，避免單位錯誤。
    @staticmethod
    def _pct_html(value: Any, as_fraction: bool = False, strong_at: float | None = None, na: str = "—") -> Raw:
        v = EmailAgent._safe_float(value)
        if v is None:
            return Raw(html.escape(na))
        if as_fraction:
            v = v * 100.0
        cls = "pos" if v > 0 else ("neg" if v < 0 else "flat")
        text = html.escape(f"{v:+.2f}%")
        tag = "strong" if (strong_at is not None and abs(v) >= strong_at) else "span"
        return Raw(f"<{tag} class='{cls}'>{text}</{tag}>")

    @staticmethod
    def _int_html(value: Any, na: str = "—") -> Raw:
        v = EmailAgent._safe_float(value)
        if v is None:
            return Raw(html.escape(na))
        cls = "pos" if v > 0 else ("neg" if v < 0 else "flat")
        text = html.escape(f"{v:+,.0f}")
        return Raw(f"<span class='{cls}'>{text}</span>")

    @staticmethod
    def _limit_label(value: Any, as_fraction: bool = False) -> tuple[str, str]:
        """依漲跌幅判斷漲跌停標籤，回傳 (方向 up/down, 文字)；不需標示時回傳 ("", "")。

        台股單日漲跌幅限制為 ±10%，但實際漲停價會依升降單位無條件捨去，
        所以常見 +9.94%、-9.93% 這類數字。這裡只用漲跌幅推估、沒有讀取實際漲跌停價：
        ≥9.9% 標「漲停／跌停」，介於 9.5%～9.9% 只標「近漲停／近跌停」，避免誤報。
        """
        v = EmailAgent._safe_float(value)
        if v is None:
            return "", ""
        if as_fraction:
            v = v * 100.0
        if v >= 9.9:
            return "up", "漲停"
        if v >= 9.5:
            return "up", "近漲停"
        if v <= -9.9:
            return "down", "跌停"
        if v <= -9.5:
            return "down", "近跌停"
        return "", ""

    @staticmethod
    def _limit_tag(value: Any, as_fraction: bool = False) -> str:
        """接近台股 ±10% 漲跌停時加上小標籤，方便一眼看到當日最極端的標的。"""
        direction, text = EmailAgent._limit_label(value, as_fraction)
        if not direction:
            return ""
        return f" <span class='tag-limit {direction}'>{text}</span>"

    @staticmethod
    def _limit_tag_text(value: Any) -> str:
        _, text = EmailAgent._limit_label(value)
        return f"（{text}）" if text else ""

    @staticmethod
    def _fmt_yi(value: Any, na: str = "—") -> str:
        """財報金額原始單位為千元（MOPS/TWSE 慣例），換算成「億元」方便閱讀。"""
        v = EmailAgent._safe_float(value)
        if v is None:
            return na
        yi = v * 1000.0 / 100_000_000.0
        return f"{yi:,.2f} 億"

    @staticmethod
    def _fmt_score(value: Any, na: str = "—") -> str:
        v = EmailAgent._safe_float(value)
        if v is None:
            return na
        return f"{v:.1f}/100"

    @staticmethod
    def _impact_badge(label: Any) -> Raw:
        text = str(label or "中性／待確認")
        if "利多" in text:
            cls = "badge-pos"
        elif "利空" in text:
            cls = "badge-neg"
        else:
            cls = "badge-neu"
        return Raw(f"<span class='badge {cls}'>{html.escape(text)}</span>")

    @staticmethod
    def _is_attention(text: Any) -> bool:
        t = str(text or "")
        if any(k in t for k in ("無需", "不需", "正常", "良好")):
            return False
        return any(k in t for k in ("重新訓練", "檢查", "異常", "警示"))

    @staticmethod
    def _co_label(name: Any, symbol: Any) -> str:
        """公司顯示名稱：有名稱顯示「名稱(代號)」；缺名稱或名稱等於代號時只顯示代號，
        避免出現「 (6582)」或「6582(6582)」。"""
        n = str(name or "").strip()
        s = str(symbol or "").strip()
        if n and n != s:
            return f"{n}({s})"
        return s

    @staticmethod
    def _short_date(value: Any) -> str:
        """RSS 的 RFC-822 日期（例如 'Mon, 11 May 2026 07:00:00 GMT'）轉成台北時間 YYYY-MM-DD；失敗回傳空字串。"""
        if not value:
            return ""
        try:
            dt = parsedate_to_datetime(str(value))
            return dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
        except Exception:
            return ""

    def _news_date_note(self, published: Any, report_date: Any) -> str:
        """在新聞旁標示發布日期；距報告日 30 天以上的舊聞用橘色提醒，避免把舊消息誤當成今日原因。"""
        d = self._short_date(published)
        if not d:
            return ""
        note = f"新聞日期 {d}"
        try:
            age = (datetime.strptime(str(report_date)[:10], "%Y-%m-%d") - datetime.strptime(d, "%Y-%m-%d")).days
            if age >= 30:
                return f"｜<span class='stale'>{note}（{age} 天前，時效偏舊）</span>"
        except Exception:
            pass
        return f"｜<span class='muted'>{note}</span>"

    # ------------------------- Qwen 文字：結論優先 --------------------------
    @classmethod
    def _split_conclusion(cls, text: str) -> tuple[str, str | None]:
        """把本機 LLM 產生的文字切成『前段推理』與『結論』，讓 Email 優先顯示結論。

        本機 Qwen3 8B 常見兩種寫法：「總結：」／「結論」單獨成行，或是「**總結：**
        接著直接寫結論句」。找不到明確標記時回傳 (原文, None)，呼叫端會整段照原樣顯示，
        不會遺漏任何內容。
        """
        if not text:
            return text, None
        lines = text.split("\n")

        # 第一輪：整行幾乎只有「總結」／「結論」這類標記本身（由下往上找，優先取最後一個）。
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            bare = re.sub(r"^[#>\-\d\.\s]+", "", line.strip())
            bare = bare.replace("*", "").strip().rstrip("：:").strip()
            if bare in cls._CONCLUSION_WORDS or (len(bare) <= 6 and any(w in bare for w in cls._CONCLUSION_WORDS)):
                before = "\n".join(lines[:i]).strip()
                after = "\n".join(lines[i + 1:]).strip()
                if after:
                    return before, after

        # 第二輪：標記與結論文字同一行（例如「**總結：** 內容...」），取最後一次出現。
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            for w in cls._CONCLUSION_WORDS:
                idx = line.find(w)
                if idx == -1:
                    continue
                remainder = line[idx + len(w):].lstrip("*").lstrip("：:").lstrip("*").strip()
                if len(remainder) >= 6:
                    before = "\n".join(lines[:i]).strip()
                    rest = "\n".join(lines[i + 1:]).strip()
                    after = remainder + (("\n" + rest) if rest else "")
                    return before, after
        return text, None

    @staticmethod
    def _markdown_lite(text: str) -> str:
        """把 **粗體**、# 標題等簡單 Markdown 轉成安全 HTML，其餘文字逐字跳脫。"""
        escaped = html.escape(text or "")
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        out_lines = []
        for line in escaped.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                out_lines.append(f"<strong>{stripped.lstrip('#').strip()}</strong>")
            else:
                out_lines.append(line)
        return "<br>".join(out_lines)

    def _qwen_block(self, content: str) -> str:
        """本機 LLM 文字：優先顯示結論。完整推理預設不放進信件（避免整段思考過程佔版面），
        EMAIL_SHOW_LLM_REASONING=true 時才以可展開區塊附上。"""
        content = (content or "").strip()
        if not content:
            return "<p class='muted'>未產生本機 Qwen3 8B 摘要。</p>"
        show_all = bool(getattr(self.config, "show_llm_reasoning", False))
        before, conclusion = self._split_conclusion(content)
        if conclusion:
            body = f"<div class='qwen-conclusion'>{self._markdown_lite(conclusion)}</div>"
            if before.strip():
                if show_all:
                    body += (
                        "<details class='reasoning'><summary>展開完整分析過程</summary>"
                        f"<div class='qwen'>{self._markdown_lite(before)}</div></details>"
                    )
                else:
                    body += (
                        "<p class='muted'>完整推理過程未放入信件；可查看附件 JSON，"
                        "或在 .env 設定 EMAIL_SHOW_LLM_REASONING=true 於信中展開。</p>"
                    )
            return body
        limit = 1500
        if show_all or len(content) <= limit:
            return f"<div class='qwen'>{self._markdown_lite(content)}</div>"
        return (
            f"<div class='qwen'>{self._markdown_lite(content[:limit])}…</div>"
            "<p class='muted'>內容較長，已截斷；全文請見附件 JSON。</p>"
        )

    # ------------------------- 代號 → 名稱 對照 -----------------------------
    def _symbol_name_map(self, report: dict[str, Any]) -> dict[str, str]:
        """從報告中已經帶名稱的區塊（漲跌股、法說會、題材、選取標的）反查代號對應的公司名稱，
        讓財報表、法說會清單也能顯示公司名稱而不只是代號。名稱等於代號者視為沒有名稱。"""
        names: dict[str, str] = {}

        def add(symbol: Any, name: Any) -> None:
            sym = str(symbol or "").strip()
            nm = str(name or "").strip()
            if sym and nm and nm != sym:
                names[sym] = nm

        mm = report.get("market_movers", {}) or {}
        for x in mm.get("movers", []) or []:
            if isinstance(x, dict):
                add(x.get("symbol"), x.get("name"))
        for x in report.get("earnings_calls", []) or []:
            if isinstance(x, dict):
                add(x.get("symbol"), x.get("name"))
        for t in report.get("bullish_themes", []) or []:
            for c in (t.get("affected_companies") or []):
                if isinstance(c, dict):
                    add(c.get("symbol"), c.get("name"))
        sel = report.get("post_market_selected")
        if isinstance(sel, dict):
            add(sel.get("symbol"), sel.get("name"))
        return names

    # ----------------------------- 各區塊 ------------------------------------
    def _executive_snapshot(self, report: dict[str, Any]) -> dict[str, Any]:
        """建立 Email 第一眼就能讀懂的盤後重點：依「正向／負向／待追蹤」分組，
        不再依資料種類（漲跌股／法說會／題材…）分組——同一件事只講一次，且好消息、
        壞消息、需要留意的事一眼就能分開看。"""
        mm = report.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
        up = sorted((x for x in movers if self._safe_float(x.get("today_change_percent"), 0) > 0),
                    key=lambda x: self._safe_float(x.get("today_change_percent"), 0), reverse=True)
        down = sorted((x for x in movers if self._safe_float(x.get("today_change_percent"), 0) < 0),
                      key=lambda x: self._safe_float(x.get("today_change_percent"), 0))
        industries = [x for x in mm.get("industry_summary", []) if isinstance(x, dict)]
        rising_ind = sorted(industries, key=lambda x: self._safe_float(x.get("today_avg_change"), -999), reverse=True)[:3]
        falling_ind = sorted(industries, key=lambda x: self._safe_float(x.get("today_avg_change"), 999))[:3]

        themes = sorted([x for x in report.get("bullish_themes", []) if isinstance(x, dict)],
                         key=lambda x: self._safe_float(x.get("strength"), 0), reverse=True)
        fin = [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)]
        fin_yoy = sorted(fin, key=lambda x: self._safe_float(x.get("revenue_yoy"), -999), reverse=True)

        calls = [x for x in report.get("earnings_calls", []) if isinstance(x, dict) and not x.get("error")]
        calls_with_risk = [x for x in calls if x.get("memo_opened") and x.get("negative_factors")]
        calls_with_risk.sort(key=lambda x: str(x.get("event_date", "")), reverse=True)

        return {
            "market_status": report.get("market_status", ""),
            "up": up[:5],
            "down": down[:5],
            "rising_industries": rising_ind,
            "falling_industries": falling_ind,
            "themes_top": themes[:3],
            "financial_growth": fin_yoy[:10],
            "calls_risk": calls_with_risk[:5],
            "research_conclusion": report.get("research_conclusion", ""),
        }

    def _theme_for_symbol(self, symbol: Any, themes: list[dict[str, Any]]) -> str:
        """個股是否掛在某個利多題材下；有的話回傳題材名稱，讓漲跌表的「相關題材」欄
        顯示具體題材（例如 PCB、記憶體），而不是重複「產業」欄已經有的官方產業分類。"""
        sym = str(symbol or "")
        for t in themes:
            for c in (t.get("affected_companies") or []):
                if str(c.get("symbol", "")) == sym:
                    return str(t.get("theme", ""))
        return ""

    def _executive_html(self, report: dict[str, Any]) -> str:
        s = self._executive_snapshot(report)
        name_map = self._symbol_name_map(report)

        def co(x: dict[str, Any]) -> str:
            sym = str(x.get("symbol", ""))
            return self._co_label(x.get("name") or name_map.get(sym), sym)

        def mover_item(x: dict[str, Any]) -> str:
            pct = x.get("today_change_percent")
            return f"<li>{self._esc(co(x))} 今日 {self._pct_html(pct, strong_at=5)}{self._limit_tag(pct)}</li>"

        def theme_item(t: dict[str, Any]) -> str:
            return f"<li>題材：{self._esc(t.get('theme',''))}（強度 <strong class='pos'>{self._fmt_score(t.get('strength'))}</strong>）</li>"

        def industry_item(x: dict[str, Any]) -> str:
            return f"<li>{self._esc(x.get('industry_name',''))} {self._pct_html(x.get('today_avg_change'), strong_at=5)}</li>"

        def fin_item(f: dict[str, Any]) -> str:
            sym = str(f.get("symbol", ""))
            label = f"{name_map[sym]}({sym})" if sym in name_map else sym
            eps_v = self._safe_float(f.get("eps"))
            eps_text = f"{eps_v:.2f}" if eps_v is not None else "—"
            return (
                f"<li><strong>{self._esc(label)}</strong>｜最新月營收 YoY "
                f"{self._pct_html(f.get('revenue_yoy'), strong_at=50)}｜EPS {eps_text}</li>"
            )

        positive_items = "".join(mover_item(x) for x in s["up"]) + "".join(theme_item(t) for t in s["themes_top"][:2])
        negative_items = "".join(mover_item(x) for x in s["down"])

        watch_items: list[str] = []
        conclusion = s["research_conclusion"]
        if conclusion and self._is_attention(conclusion):
            watch_items.append(f"<li>模型狀態：<span class='warn'>{self._esc(conclusion)}</span></li>")
        for x in s["calls_risk"]:
            sym = str(x.get("symbol", ""))
            factors = "；".join(str(f) for f in (x.get("negative_factors") or [])[:2])
            if factors:
                watch_items.append(f"<li><strong>{self._esc(sym)}</strong> 法說風險：<span class='warn'>{self._esc(factors)}</span></li>")
        for t in s["themes_top"][:1]:
            flags = [f for f in (t.get("contradiction_flags") or []) if f]
            if flags:
                watch_items.append(f"<li>{self._esc(t.get('theme',''))}：<span class='warn'>{self._esc(flags[0])}</span></li>")

        industry_items = "".join(industry_item(x) for x in s["rising_industries"]) + \
            "".join(industry_item(x) for x in s["falling_industries"])
        theme_list_items = "".join(
            f"<li><strong>{self._esc(t.get('theme',''))}</strong>｜{self._fmt_score(t.get('strength'))}｜{self._esc(t.get('status',''))}</li>"
            for t in s["themes_top"]
        ) or "<li>目前沒有通過證據門檻的利多題材</li>"
        fin_items = "".join(fin_item(f) for f in s["financial_growth"]) or "<li>目前沒有可用財報資料。</li>"

        return f"""
<section class='executive'>
<h2>① 今日 AI 研究重點</h2>
<div class='notice'>{self._esc(s['market_status'])}</div>
<p class='legend'><span class='pos'>■</span>紅字＝上漲／利多　<span class='neg'>■</span>綠字＝下跌／利空　<span class='warn'>■</span>橘字＝待追蹤</p>
<h3>正向訊號</h3>
<ul>{positive_items or "<li>今日無明顯正向訊號</li>"}</ul>
<h3>負向訊號</h3>
<ul>{negative_items or "<li>今日無明顯負向訊號</li>"}</ul>
<h3>待追蹤／待查證</h3>
<ul>{"".join(watch_items) or "<li>目前沒有需要特別留意的項目</li>"}</ul>
<h3>② 產業強弱</h3>
<ul>{industry_items or "<li>今日無明顯產業</li>"}</ul>
<h3>③ 目前主要利多題材</h3>
<ul>{theme_list_items}</ul>
<h3>④ 營收成長重點</h3>
<ul>{fin_items}</ul>
</section>
"""

    def _market_section(self, report: dict[str, Any]) -> str:
        mm = report.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
        up_list = sorted((x for x in movers if self._safe_float(x.get("today_change_percent"), 0) > 0),
                          key=lambda x: self._safe_float(x.get("today_change_percent"), 0), reverse=True)[:5]
        down_list = sorted((x for x in movers if self._safe_float(x.get("today_change_percent"), 0) < 0),
                            key=lambda x: self._safe_float(x.get("today_change_percent"), 0))[:5]
        industry = [x for x in mm.get("industry_summary", []) if isinstance(x, dict)]
        industry = sorted(industry, key=lambda x: self._safe_float(x.get("today_avg_change"), -999), reverse=True)[:6]
        themes = [x for x in report.get("bullish_themes", []) if isinstance(x, dict)]

        def mover_row(x: dict[str, Any]) -> list[Any]:
            pct = x.get("today_change_percent")
            theme_name = self._theme_for_symbol(x.get("symbol"), themes)
            theme_cell: Any = theme_name if theme_name else Raw("<span class='muted'>—</span>")
            return [
                f"{x.get('name', '')} ({x.get('symbol', '')})",
                x.get("industry_name", ""),
                Raw(f"{self._pct_html(pct, strong_at=5)}{self._limit_tag(pct)}"),
                self._pct_html(x.get("ret_5d")),
                theme_cell,
            ]

        headers = ["公司", "產業", "今日", "5日", "相關題材"]
        up_table = self._table(headers, [mover_row(x) for x in up_list])
        down_table = self._table(headers, [mover_row(x) for x in down_list])

        def industry_row(x: dict[str, Any]) -> list[Any]:
            return [
                f"{x.get('industry_code', '')} {x.get('industry_name', '')}",
                x.get("candidate_count", ""),
                self._pct_html(x.get("today_avg_change"), strong_at=5),
                self._pct_html(x.get("ret_5d_avg")),
                f"{x.get('rising_count', 0)} / {x.get('falling_count', 0)}",
            ]

        industry_table = self._table(["產業", "樣本", "今日平均", "5日平均", "漲/跌"], [industry_row(x) for x in industry])

        return f"""
<section>
<h2>一、今日市場重點</h2>
<p class='muted'>紅字＝上漲／利多　綠字＝下跌／利空　橘字＝待追蹤。「相關題材」欄只在個股掛在下方利多題材下時才會顯示題材名稱；
沒有對應題材時留白，一般產業分類請見「產業」欄，不重複寫「缺乏新聞、無法歸因」這類制式文字。</p>
<h3>漲幅重點</h3>
{up_table}
<h3>跌幅重點</h3>
{down_table}
<h3>產業強弱 Top 6</h3>
{industry_table}
</section>
"""

    def _earnings_section(
        self, report: dict[str, Any], heading: str = "法人說明會／法說會", limit: int = 15,
    ) -> str:
        calls = [x for x in report.get("earnings_calls", []) if isinstance(x, dict) and not x.get("error")]
        calls = sorted(calls, key=lambda x: str(x.get("event_date", "")), reverse=True)[:limit]
        name_map = self._symbol_name_map(report)
        evidences = {str(x.get("evidence_level", "")) for x in calls}
        uniform_evidence = len(evidences) == 1
        has_5d = any(x.get("post_5d_return") is not None for x in calls)

        rows: list[list[Any]] = []
        details: list[str] = []
        for x in calls:
            memo_ok = bool(x.get("memo_opened") and x.get("summary"))
            sym = str(x.get("symbol", ""))
            date = str(x.get("event_date", ""))
            label = self._co_label(x.get("name") or name_map.get(sym), sym)
            conf = self._safe_float(x.get("confidence"))
            badge = self._impact_badge(x.get("impact")) if memo_ok else Raw("<span class='badge badge-neu'>無Fugle備忘錄摘要</span>")
            row: list[Any] = [date, label, badge, f"{conf:.0f}%" if (memo_ok and conf is not None) else "—"]
            if not uniform_evidence:
                row.append(x.get("evidence_level", ""))
            row.append(self._pct_html(x.get("post_1d_return"), as_fraction=True))
            if has_5d:
                row.append(self._pct_html(x.get("post_5d_return"), as_fraction=True))
            rows.append(row)
            if not memo_ok:
                continue

            lead = f"<p>{self._esc((x.get('one_line_summary') or x.get('summary') or '')[:2200])}</p>"
            pos = "、".join(x.get("positive_terms", [])[:8]) or "—"
            neg = "、".join(x.get("negative_terms", [])[:8]) or "—"
            src = x.get("source_url", "")
            details.append(
                f"<div class='theme'><h3>{self._esc(date)}｜{self._esc(label)}　{self._impact_badge(x.get('impact'))}</h3>"
                f"{lead}"
                f"<p><strong>財務：</strong>{self._esc('；'.join(x.get('financial_highlights',[])[:6]))}</p>"
                f"<p><strong>展望：</strong>{self._esc('；'.join(x.get('guidance',[])[:6]))}</p>"
                f"<p><strong class='pos'>利多：</strong>{self._esc('、'.join(x.get('positive_factors',[])[:8]) or pos)}</p>"
                f"<p><strong class='neg'>利空/風險：</strong>{self._esc('、'.join(x.get('negative_factors',[])[:8]) or neg)}</p>"
                f"<p><a href='{self._esc(src)}'>來源：富果法說會備忘錄</a></p></div>"
            )

        headers = ["日期", "公司", "法說訊號", "信心"] + ([] if uniform_evidence else ["證據"]) + ["後1日"] + (["後5日"] if has_5d else [])
        calls_table = self._table(headers, rows)
        evidence_note = ""
        if uniform_evidence and evidences != {""}:
            evidence_note = f"證據等級：{self._esc(next(iter(evidences)))}（全部相同）。"
        return f"""
<section>
<h2>{self._esc(heading)}</h2>
<p class='muted'>法說會摘要必須來自 AI Agent 實際開啟的 Fugle「法說會備忘錄」詳細文章（由本機 Ollama Agent 整理）；MOPS/TWSE/TPEx 僅作日期補漏。{evidence_note}
「後1日／後5日」是法說後實際股價反應，「—」代表交易日尚不足、無法計算。</p>
{calls_table}
{''.join(details)}
</section>
"""

    _COUNT_DRIVER = re.compile(r"^\s*(\d+\s*家|發現\s*\d+\s*筆|跨來源佐證累計\s*\d+)")

    def _themes_section(self, report: dict[str, Any]) -> str:
        """題材雷達：只留題材／強度／狀態，逐題材的證據清單與查證提醒不在這裡重複——
        最重要的一兩個題材的查證提醒已經在最上方「待追蹤／待查證」出現過一次。"""
        themes = [x for x in report.get("bullish_themes", []) if isinstance(x, dict)]
        themes = sorted(themes, key=lambda x: self._safe_float(x.get("strength"), 0), reverse=True)[:8]

        def theme_row(t: dict[str, Any]) -> list[Any]:
            strength_val = self._safe_float(t.get("strength"), 0)
            score_text = self._fmt_score(t.get("strength"))
            strength_cell: Any = Raw(f"<strong class='pos'>{score_text}</strong>") if strength_val >= 50 else score_text
            return [t.get("theme", ""), strength_cell, t.get("status", "")]

        themes_table = self._table(["題材", "強度", "狀態"], [theme_row(t) for t in themes])
        return f"""
<section>
<h2>二、目前主要利多題材</h2>
<p class='muted'>以下為研究引擎依新聞、產業鏈、接單／出貨與營收支撐計算出的主要題材；強度 50 以上以粗體紅字標示。
Agent 另外負責查找近期市場原因與補充證據，最新查證狀態請見最上方「待追蹤／待查證」與最下方「資料來源與查證」。</p>
{themes_table}
</section>
"""

    def _financial_section(self, report: dict[str, Any], name_map: dict[str, str] | None = None) -> str:
        name_map = name_map or {}
        items = [x for x in (report.get("financial_snapshots", []) or []) if isinstance(x, dict)]
        items = sorted(items, key=lambda x: self._safe_float(x.get("revenue_yoy"), -999), reverse=True)[:10]
        rows = []
        for x in items:
            symbol = str(x.get("symbol", ""))
            label = f"{name_map[symbol]}({symbol})" if symbol in name_map else symbol
            eps_val = self._safe_float(x.get("eps"))
            rows.append([
                label,
                self._fmt_yi(x.get("latest_month_revenue")),
                self._pct_html(x.get("revenue_yoy"), strong_at=50),
                f"{eps_val:.2f}" if eps_val is not None else "—",
            ])
        table_html = self._table(["公司", "最新月營收", "YoY", "EPS"], rows)
        return f"""
<section>
<h2>三、營收成長重點</h2>
<p class='muted'>正文列出目前最值得注意的營收 YoY／EPS 變化（依 YoY 由高到低取前 10 名）；完整財報數據（含營業利益、淨利）保留於附件。
金額已換算為「億元」，年增率 50% 以上以粗體紅字標示。</p>
{table_html}
</section>
"""

    def _qwen_section(self, report: dict[str, Any]) -> str:
        llm = report.get("ollama_analysis", {}) or {}
        content = llm.get("content") or llm.get("error") or ""
        body = self._qwen_block(content)
        return f"""
<section>
<h2>四、本機 Qwen3 8B 研究摘要</h2>
<p class='muted'>本機 8B 模型的整理僅供參考，可能有公司或數字對應錯誤，請以上方各節的原始數字與公告為準。</p>
{body}
</section>
"""

    def _selected_section(self, report: dict[str, Any]) -> str:
        selected = report.get("post_market_selected")
        if not isinstance(selected, dict) or not selected:
            return ""
        symbol = selected.get("symbol", "")
        name = selected.get("name", symbol)
        error = selected.get("error")
        if error:
            return (
                "<section><h2>五、最後選取標的盤後事件歸因</h2>"
                f"<p>目前選取標的 {self._esc(name)}({self._esc(symbol)}) 的盤後事件歸因未完成：{self._esc(error)}</p></section>"
            )

        price = selected.get("price_context", {}) or {}
        inst = selected.get("institution_context", {}) or {}
        fundamentals = selected.get("fundamentals", {}) or {}
        reasons = [r for r in (selected.get("reason_candidates") or []) if isinstance(r, dict)][:5]
        news = [n for n in (selected.get("verified_news") or []) if isinstance(n, dict)]

        cur_price = self._safe_float(price.get("current_price"))
        chg_pct = price.get("today_change_percent")
        rel_taiex = price.get("relative_to_taiex_pct")
        taiex = price.get("taiex", {}) or {}

        price_line = f"現價 <strong>{cur_price:.2f}</strong> " if cur_price is not None else "現價 — "
        price_line += (
            f"{self._pct_html(chg_pct, strong_at=5)}{self._limit_tag(chg_pct)}"
            f"｜相對大盤 {self._pct_html(rel_taiex, strong_at=3)}"
            f"（加權指數當日 {self._pct_html(taiex.get('change_pct'), strong_at=1)}）"
        )

        inst_lines = []
        if inst.get("available"):
            latest = inst.get("latest", {}) or {}
            totals = inst.get("totals_10d", {}) or {}
            inst_lines.append(
                f"最近一日三大法人合計 {self._int_html(latest.get('total_net'))} 股"
                f"（外資 {self._int_html(latest.get('foreign_net'))}）"
            )
            inst_lines.append(
                f"近10日合計 {self._int_html(totals.get('total_net'))} 股"
                f"（外資 {self._int_html(totals.get('foreign_net'))}｜投信 {self._int_html(totals.get('trust_net'))}）"
            )
        else:
            inst_lines.append("目前沒有可用的法人交易資料。")

        rev_yoy = fundamentals.get("revenue_yoy")
        eps_val = self._safe_float(fundamentals.get("eps"))
        fin_line = (
            f"最新月營收 {self._fmt_yi(fundamentals.get('latest_month_revenue'))}"
            f"（YoY {self._pct_html(rev_yoy, strong_at=30)}）"
            f"｜最新 EPS {f'{eps_val:.2f}' if eps_val is not None else '—'} 元"
        )

        report_date = report.get("report_date", "")
        pub_by_link = {n.get("link"): n.get("published") for n in news if n.get("link")}

        reason_items = "".join(
            f"<li><strong>{self._esc(r.get('factor',''))}</strong>｜"
            f"{self._impact_badge(r.get('direction'))}｜{self._esc(r.get('evidence',''))}"
            f"{self._news_date_note(pub_by_link.get(r.get('link')), report_date) if r.get('link') else ''}</li>"
            for r in reasons
        ) or "<li>目前沒有明顯的量化驅動因素。</li>"

        # 「可能驅動因素」多半就是同一批新聞，這裡排除已列在上方的項目，避免同一則新聞出現兩次；
        # 相關度相同時新的排前面。
        shown_links = {r.get("link") for r in reasons if r.get("link")}
        kept_news = [
            n for n in news
            if not str(n.get("verification", "")).startswith("低業務相關度") and n.get("link") not in shown_links
        ]
        kept_news = sorted(
            kept_news,
            key=lambda n: (self._safe_float(n.get("business_relevance"), 0), self._short_date(n.get("published"))),
            reverse=True,
        )[:4]
        news_items = "".join(
            f"<li><a href='{self._esc(n.get('link',''))}'>{self._esc(n.get('title',''))}</a>"
            f"｜{self._esc(n.get('source',''))}｜{self._esc(n.get('verification',''))}"
            f"{self._news_date_note(n.get('published'), report_date)}</li>"
            for n in kept_news
        ) or "<li>目前沒有通過相關度門檻的新聞。</li>"

        llm = selected.get("ollama_analysis", {}) or {}
        ai_block = self._qwen_block(llm.get("content") or llm.get("error") or "")

        return f"""
<section>
<h2>五、最後選取標的盤後事件歸因｜{self._esc(name)}({self._esc(symbol)})</h2>
<p class='muted'>原始委買賣報價、逐日法人明細與 MOPS 原始欄位已收斂為以下重點；如需逐筆資料，請見 research JSON（附件或 output/research_reports）。</p>
<p class='line'>{price_line}</p>
<p class='line'>{'；'.join(inst_lines)}</p>
<p class='line'>{fin_line}</p>
<h3>可能驅動因素（依證據強度排序）</h3>
<ul>{reason_items}</ul>
<h3>其他相關新聞（已濾除低相關度與上方已列項目）</h3>
<ul>{news_items}</ul>
<h3>AI 個股歸因分析</h3>
{ai_block}
</section>
"""

    def _cnyes_section(self, news: dict[str, Any] | None, heading: str = "近 N 日重大新聞（鉅亨網頭條）") -> str:
        """近 N 日財經新聞摘要（鉅亨網頭條，見 cnyes_news_agent.py）。`news`是
        `CnyesNewsAgent.build_digest()` 回傳的那包字典；沒有資料時回傳空字串，
        信件排版與章節編號不受影響（目前只有早報會呼叫這個方法）。"""
        if not isinstance(news, dict) or not news:
            return ""
        digest = news.get("digest", {}) or {}
        days = news.get("window_days", 2)
        total = news.get("total_in_window", 0)
        source_url = news.get("source_url", "https://news.cnyes.com/news/cat/headline")

        if not digest.get("available"):
            note = self._esc(digest.get("error") or "本機 Qwen3 8B 目前無法使用，暫時沒有新聞摘要。")
            return f"""
<section>
<h2>{self._esc(heading)}</h2>
<p class='muted'>{note}共擷取到 {total} 則候選新聞，原始清單請見附件 JSON。</p>
</section>
"""

        items = digest.get("items", []) or []
        overall = digest.get("overall_take", "")
        overall_html = f"<div class='callout-info'>{self._esc(overall)}</div>" if overall else ""

        def item_html(it: dict[str, Any]) -> str:
            sources = it.get("sources", []) or []
            src_links = "、".join(
                f"<a href='{self._esc(s.get('url',''))}'>{self._esc(str(s.get('title',''))[:26])}</a>" for s in sources[:3]
            ) or "—"
            kw = "、".join(it.get("keywords", [])[:6])
            kw_html = f"<br><span class='sub'>關鍵字：{self._esc(kw)}</span>" if kw else ""
            return (
                "<div class='theme'>"
                f"<strong>{self._esc(it.get('headline',''))}</strong>　{self._impact_badge(it.get('impact'))}　"
                f"<span class='muted'>{self._esc(it.get('relevance',''))}</span>"
                f"<p>{self._esc(it.get('summary',''))}</p>"
                f"<p class='muted'>資料來源：{src_links}</p>{kw_html}"
                "</div>"
            )

        body = "".join(item_html(it) for it in items) or "<p class='muted'>本機 Qwen3 8B 判斷這段期間沒有對台股特別重要的新聞。</p>"
        return f"""
<section>
<h2>{self._esc(heading)}</h2>
<p class='muted'>來源：<a href='{self._esc(source_url)}'>鉅亨網頭條新聞</a>；本機 Qwen3 8B 從 {total} 則候選新聞中挑出重點，同一事件的多篇報導已合併，不代表投資建議。</p>
{overall_html}
{body}
</section>
"""

    @staticmethod
    def _base_css() -> str:
        """兩封信（盤後報告／早報）共用的樣式；早報透過繼承呼叫同一份，改版只需要改這裡一次。"""
        return """
body{font-family:Arial,'Microsoft JhengHei',sans-serif;background:#f5f7fa;color:#202124;line-height:1.6;margin:0;padding:0}
.container{max-width:1100px;margin:0 auto;background:#fff;padding:28px 34px}
h1{margin:0 0 6px;font-size:28px}
h2{margin-top:30px;border-bottom:2px solid #e5e7eb;padding-bottom:6px}
h3{margin-top:20px}
.muted{color:#6b7280}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e5e7eb;padding:7px 8px;vertical-align:top;text-align:left}
th{background:#f3f4f6}
.table-wrap{overflow-x:auto}
.theme{border-left:4px solid #d1d5db;background:#fafafa;padding:12px 14px;margin:10px 0}
.qwen{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:14px;white-space:pre-wrap;word-break:break-word}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700}
.footer{margin-top:34px;padding-top:14px;border-top:1px solid #e5e7eb;color:#6b7280;font-size:12px}
.executive{background:#f8fafc;border:1px solid #dbeafe;border-radius:10px;padding:16px;margin:20px 0}
.executive h2{margin-top:0}
.notice{background:#eef6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 10px;margin:8px 0 14px}
.line{margin:6px 0}
.pos{color:#c0392b}
.neg{color:#1e8449}
.flat{color:#6b7280}
strong.pos,strong.neg{font-weight:800}
.badge-pos{background:#fdecea;color:#c0392b}
.badge-neg{background:#eafaf1;color:#1e8449}
.badge-neu{background:#f3f4f6;color:#4b5563}
.tag-limit{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;font-weight:700;margin-left:4px}
.tag-limit.up{background:#fdecea;color:#c0392b}
.tag-limit.down{background:#eafaf1;color:#1e8449}
.callout-info{background:#eef6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px 12px;font-weight:700}
.callout-warn{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:8px;padding:10px 12px;font-weight:700}
.qwen-conclusion{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px 14px;margin:10px 0;font-weight:600}
details.reasoning{margin:8px 0 14px}
details.reasoning summary{cursor:pointer;color:#2563eb;font-weight:700;margin-bottom:6px}
.stale,.warn{color:#b45309;font-weight:700}
.legend{font-size:12px;color:#6b7280;margin:4px 0 14px}
.legend span{font-weight:700}
.sub{color:#4b5563;font-size:13px;font-weight:400}
"""

    def build_html(self, report: dict[str, Any]) -> str:
        report_date = self._esc(report.get("report_date", ""))
        watchlist = report.get("research_scope", {}).get("watchlist_symbols", []) or []
        sources = report.get("official_sources", {}) or {}
        name_map = self._symbol_name_map(report)
        source_text = (
            f"TWSE公告 {sources.get('twse_announcements_rows', 0)} 筆、"
            f"TPEx公告 {sources.get('tpex_announcements_rows', 0)} 筆、"
            f"TWSE營收 {sources.get('twse_revenue_rows', 0)} 筆、"
            f"TPEx營收 {sources.get('tpex_revenue_rows', 0)} 筆、"
            f"TWSE財報 {sources.get('twse_income_rows', 0)} 筆、"
            f"TPEx財報 {sources.get('tpex_income_rows', 0)} 筆"
        )
        cnyes_news = report.get("cnyes_news")
        cnyes_note = ""
        if isinstance(cnyes_news, dict) and cnyes_news:
            source_text += f"、鉅亨近兩日新聞 {cnyes_news.get('total_in_window', 0)} 篇"
            cnyes_note = (
                "<p class='muted'>本次另以鉅亨網頭條近兩日新聞作為近期市場研究資料庫，由 Agent 自主搜尋與整理；"
                "新聞屬市場線索，重大事項仍需回看原始公告。</p>"
            )
        return f"""<!doctype html>
<html lang='zh-Hant'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
{self._base_css()}
</style>
</head>
<body>
<div class='container'>
<h1>AI 台股盤後財報</h1>
<div class='muted'>研究日期：{report_date}｜本機 Qwen3 8B + Fugle / TWSE / TPEx / MOPS 證據整合</div>
<p><strong>研究自選股：</strong>{self._esc('、'.join(map(str, watchlist)))}</p>
{self._executive_html(report)}
{self._market_section(report)}
{self._themes_section(report)}
{self._financial_section(report, name_map)}
{self._qwen_section(report)}
{self._selected_section(report)}
<section>
<h2>六、資料來源與查證</h2>
<p>{self._esc(source_text)}</p>
<p class='muted'>法說會、重大訊息與財報/營收以公開原始資料為優先；新聞與題材僅作研究線索。未完成官方交叉驗證者標示為待查證。</p>
{cnyes_note}
<p class='muted'>完整 research JSON / Markdown 已附加於本信；需要更完整證據時請查看附件與原始來源。</p>
</section>
<div class='footer'>
本報告是盤後研究與模型治理工具，不保證新聞真偽，也不代表投資建議。重大消息請回看 MOPS/TWSE/TPEx 原始公告。
</div>
</div>
</body>
</html>"""

    def build_text(self, report: dict[str, Any]) -> str:
        """純文字備援版：先給結論，再附必要細節。"""
        s = self._executive_snapshot(report)
        lines = [
            f"AI 台股盤後財報｜{report.get('report_date','')}",
            f"市場狀態：{s['market_status']}",
            "",
            "====================",
            "📌 今日先看這裡",
            "====================",
        ]
        lines.append("1. 上漲重點")
        for x in s["up"]:
            pct = x.get("today_change_percent")
            lines.append(
                f"- {x.get('name','')}({x.get('symbol','')}) {self._safe_float(pct,0):+.2f}%{self._limit_tag_text(pct)}"
                f"｜{x.get('industry_name','')}｜{x.get('reason','待查證')}"
            )
        lines.append("2. 下跌重點")
        for x in s["down"]:
            pct = x.get("today_change_percent")
            lines.append(
                f"- {x.get('name','')}({x.get('symbol','')}) {self._safe_float(pct,0):+.2f}%{self._limit_tag_text(pct)}"
                f"｜{x.get('industry_name','')}｜{x.get('reason','待查證')}"
            )
        lines.append("3. 產業強弱")
        lines.append("- 較強：" + ("；".join(f"{x.get('industry_name','')} {self._safe_float(x.get('today_avg_change'),0):+.2f}%" for x in s["rising_industries"]) or "無"))
        lines.append("- 較弱：" + ("；".join(f"{x.get('industry_name','')} {self._safe_float(x.get('today_avg_change'),0):+.2f}%" for x in s["falling_industries"]) or "無"))
        lines.append("4. 利多題材")
        for t in s["themes_top"]:
            lines.append(f"- {t.get('theme')}｜強度 {self._safe_float(t.get('strength'),0):.1f}/100｜{t.get('status')}")
        lines.append("5. 待追蹤／待查證")
        if s["research_conclusion"] and self._is_attention(s["research_conclusion"]):
            lines.append(f"- 模型狀態：{s['research_conclusion']}")
        for x in s["calls_risk"]:
            factors = "；".join(str(f) for f in (x.get("negative_factors") or [])[:2])
            if factors:
                lines.append(f"- {x.get('symbol','')} 法說風險：{factors}")
        if not (s["calls_risk"] or (s["research_conclusion"] and self._is_attention(s["research_conclusion"]))):
            lines.append("- 目前沒有需要特別留意的項目。")
        lines.append("6. 財報／營收")
        name_map = self._symbol_name_map(report)
        for f in s["financial_growth"]:
            sym = str(f.get("symbol", ""))
            label = f"{name_map[sym]}({sym})" if sym in name_map else sym
            eps_v = self._safe_float(f.get("eps"))
            eps_text = f"{eps_v:.2f}" if eps_v is not None else "—"
            lines.append(f"- {label}｜最新月營收 YoY {self._safe_float(f.get('revenue_yoy'),0):+.2f}%｜EPS {eps_text}")
        lines += ["", "====================", "詳細研究", "====================", ""]
        lines.append("【Qwen3 8B 研究結論】")
        llm = report.get("ollama_analysis", {}) or {}
        if llm.get("validation") == "failed":
            lines.append(f"- {llm.get('error','摘要未通過一致性檢查')}")
        else:
            raw_content = str(llm.get("content") or llm.get("error") or "未產生摘要。")
            _, conclusion = self._split_conclusion(raw_content)
            lines.append(conclusion or raw_content)
        lines += [
            "",
            "完整明細（含個股事件歸因）已附上 research JSON / Markdown，或請見 HTML 版本內文；"
            "最近 5 天法說會與財經新聞摘要請見早報。",
            "資料僅供盤後研究與模型治理，重大消息請回看 MOPS/TWSE/TPEx 原始公告。",
        ]
        return "\n".join(lines)

    # ----------------------------- send ------------------------------------
    def _smtp_send_once(self, message: EmailMessage, host: str, port: int, use_ssl: bool) -> list[str]:
        """執行單次 SMTP 傳送，回傳每個階段的診斷訊息。"""
        logs: list[str] = []
        context = ssl.create_default_context()
        logs.append(f"CONNECT {host}:{port} {'SSL' if use_ssl else 'STARTTLS'}")
        if use_ssl or port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
                logs.append("CONNECTED")
                code, _ = server.ehlo()
                logs.append(f"EHLO {code}")
                logs.append("LOGIN")
                server.login(self.config.sender, self.config.app_password)
                logs.append("LOGIN_OK")
                server.send_message(message)
                logs.append("SEND_OK")
        else:
            with smtplib.SMTP(host, port, timeout=60) as server:
                logs.append("CONNECTED")
                code, _ = server.ehlo()
                logs.append(f"EHLO {code}")
                logs.append("STARTTLS")
                server.starttls(context=context)
                logs.append("STARTTLS_OK")
                code, _ = server.ehlo()
                logs.append(f"EHLO_TLS {code}")
                logs.append("LOGIN")
                server.login(self.config.sender, self.config.app_password)
                logs.append("LOGIN_OK")
                server.send_message(message)
                logs.append("SEND_OK")
        return logs

    def _smtp_candidates(self) -> list[tuple[str, int, bool]]:
        """依目前設定優先嘗試，失敗後自動切換 Gmail 的另一個官方 SMTP 模式。"""
        c = self.config
        primary = (c.smtp_host, c.smtp_port, c.use_ssl or c.smtp_port == 465)
        if primary[1] == 465:
            fallback = (c.smtp_host, 587, False)
        else:
            fallback = (c.smtp_host, 465, True)
        # 去除完全相同的候選。
        return [primary] if primary == fallback else [primary, fallback]

    def _smtp_send(self, message: EmailMessage) -> list[str]:
        errors: list[str] = []
        for host, port, use_ssl in self._smtp_candidates():
            try:
                return self._smtp_send_once(message, host, port, use_ssl)
            except smtplib.SMTPAuthenticationError:
                # 驗證錯誤不因換連接埠而假裝可修復；保留原始錯誤。
                raise
            except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, smtplib.SMTPException, OSError) as exc:
                errors.append(f"{host}:{port} {type(exc).__name__}: {exc}")
        raise smtplib.SMTPServerDisconnected(" | ".join(errors) or "SMTP connection failed")

    def diagnose_smtp(self) -> dict[str, Any]:
        """只測試 SMTP 連線、TLS、登入，不寄出任何郵件。"""
        result: dict[str, Any] = {
            "sender": self.config.sender,
            "smtp_host": self.config.smtp_host,
            "candidates": [],
        }
        if not self.status()["configured"]:
            result["configured"] = False
            result["error"] = self.status()["missing"]
            return result
        result["configured"] = True
        for host, port, use_ssl in self._smtp_candidates():
            item: dict[str, Any] = {"host": host, "port": port, "use_ssl": use_ssl, "ok": False, "steps": []}
            try:
                # 空白測試訊息只在真正 send_report 時送出；這裡到 LOGIN_OK 即停止。
                context = ssl.create_default_context()
                if use_ssl or port == 465:
                    with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
                        item["steps"].append("CONNECTED")
                        code, _ = server.ehlo()
                        item["steps"].append(f"EHLO {code}")
                        server.login(self.config.sender, self.config.app_password)
                        item["steps"].append("LOGIN_OK")
                else:
                    with smtplib.SMTP(host, port, timeout=60) as server:
                        item["steps"].append("CONNECTED")
                        code, _ = server.ehlo()
                        item["steps"].append(f"EHLO {code}")
                        server.starttls(context=context)
                        item["steps"].append("STARTTLS_OK")
                        code, _ = server.ehlo()
                        item["steps"].append(f"EHLO_TLS {code}")
                        server.login(self.config.sender, self.config.app_password)
                        item["steps"].append("LOGIN_OK")
                item["ok"] = True
            except smtplib.SMTPAuthenticationError as exc:
                item["error_type"] = type(exc).__name__
                item["error"] = str(exc)
            except (smtplib.SMTPException, OSError) as exc:
                item["error_type"] = type(exc).__name__
                item["error"] = str(exc)
            result["candidates"].append(item)
        return result

    def send_report(self, report: dict[str, Any], force: bool = False) -> dict[str, Any]:
        report_date = str(report.get("report_date", ""))
        status = self.status()
        if not status["enabled"]:
            return {"sent": False, "skipped": True, "reason": "EMAIL_ENABLED=false", "status": status}
        if not status["configured"]:
            return {"sent": False, "skipped": True, "reason": f"Email 尚未設定：缺少 {', '.join(status['missing'])}", "status": status}

        # 本專案不再限制同一天只能寄送一次。
        # 每執行一次 send_report() 就實際寄送一次，方便：
        # 1. Windows 工作排程測試
        # 2. 手動重寄最新報告
        # 3. 法說會/研究內容更新後立即重新寄送
        # force 參數保留，只是為了相容舊版呼叫方式。
        subject = f"{self.config.subject_prefix}盤後報告 {report_date}｜今日訊號 × 產業強弱 × 利多利空"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = formataddr((self.config.display_name, self.config.sender))
        msg["To"] = self.config.recipient
        msg.set_content(self.build_text(report), charset="utf-8")
        msg.add_alternative(self.build_html(report), subtype="html", charset="utf-8")

        attachments: list[str] = []
        if self.config.attach_reports:
            json_path = Path(str(report.get("json_path", "")))
            md_path = Path(str(report.get("markdown_path", "")))
            # 報告路徑必須位於本專案 output/research_reports；避免誤附到其他檔案。
            allowed_root = self.output_dir.resolve()
            for p, maintype, subtype in [
                (json_path, "application", "json"),
                (md_path, "text", "markdown"),
            ]:
                try:
                    rp = p.resolve()
                    if rp.exists() and allowed_root in rp.parents:
                        data = rp.read_bytes()
                        msg.add_attachment(
                            data,
                            maintype=maintype,
                            subtype=subtype,
                            filename=rp.name,
                        )
                        attachments.append(rp.name)
                except Exception:
                    continue

        try:
            smtp_log = self._smtp_send(msg)
            result = {
                "sent": True,
                "skipped": False,
                "report_date": report_date,
                "recipient": self.config.recipient,
                "subject": subject,
                "attachments": attachments,
                "smtp_log": smtp_log,
                "status": status,
            }
            self._write_marker(report_date or "unknown", result)
            return result
        except smtplib.SMTPAuthenticationError as exc:
            return {
                "sent": False,
                "skipped": False,
                "error": f"Gmail SMTP 驗證失敗：{exc}. 請確認 2 步驟驗證與 App Password。",
                "status": status,
            }
        except (smtplib.SMTPException, OSError) as exc:
            return {
                "sent": False,
                "skipped": False,
                "error": f"SMTP 寄信失敗：{exc}",
                "status": status,
            }

    def load_latest_report(self) -> dict[str, Any] | None:
        files = sorted(self.output_dir.glob("research_*.json"))
        if not files:
            return None
        try:
            return json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            return None

    def send_latest_report(self, force: bool = False) -> dict[str, Any]:
        report = self.load_latest_report()
        if not report:
            return {"sent": False, "skipped": True, "reason": "目前沒有已保存的盤後研究報告。", "status": self.status()}
        return self.send_report(report, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 盤後財報 Email Agent")
    parser.add_argument("--status", action="store_true", help="顯示 Email Agent 設定狀態")
    parser.add_argument("--send-latest", action="store_true", help="寄送最新研究報告")
    parser.add_argument("--force", action="store_true", help="相容舊版參數；目前每次執行都會寄送")
    parser.add_argument("--diagnose-smtp", action="store_true", help="只測試 SMTP 連線/TLS/登入，不寄信")
    parser.add_argument("--preview", action="store_true", help="用最新研究報告產生 HTML 預覽檔（data/email/preview_latest.html），不寄信")
    args = parser.parse_args()

    agent = EmailAgent(Path(__file__).resolve().parent)

    if args.preview:
        latest = agent.load_latest_report()
        if not latest:
            print("目前沒有已保存的盤後研究報告，無法產生預覽。")
            raise SystemExit(1)
        preview_path = agent.email_dir / "preview_latest.html"
        preview_path.write_text(agent.build_html(latest), encoding="utf-8")
        print(f"已產生預覽：{preview_path}")
        raise SystemExit(0)

    if args.diagnose_smtp:
        print(json.dumps(agent.diagnose_smtp(), ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0)

    if args.status or not args.send_latest:
        print(json.dumps(agent.status(), ensure_ascii=False, indent=2))

    if args.send_latest:
        result = agent.send_latest_report(force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0 if result.get("sent") or result.get("skipped") else 1)
