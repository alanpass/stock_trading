# -*- coding: utf-8 -*-
"""
AI 台股盤後財報 Email Agent v4

設計目標
============================================================
1. 不把完整 ResearchAgent 報告整份塞進 Email 本文。
2. 先用本機 Qwen3 8B 擔任「盤後研究摘要 Agent」，把已取得的市場、
   財報、法說會、題材與模型證據整合成使用者容易閱讀的重點。
3. Email 本文固定保留研究結果需要的區塊：
      - 今日市場重點
      - 最近 5 天法說會（不在 Agent 摘要重複）
      - Agent 研究筆記
      - 目前主要利多題材
      - 營收成長重點（若結構化資料不足，由 Agent 先查新聞再補充）
4. 完整 research JSON / Markdown 仍作為附件，不重複塞進正文。
5. 絕不把 Qwen3 的思考過程寫進 Email。
6. 上漲 / 利多使用紅字；下跌 / 利空使用綠字；待追蹤使用橘色。
7. 不再限制同一天只能寄一次。每執行 send_report() 就寄一次。
8. 保留 --force 參數以相容舊版測試指令。

注意
============================================================
本程式只負責呈現 ResearchAgent / Research Assistant Agent 已取得的證據。
若 Agent 主動使用即時新聞工具，Email 可呈現其研究結果與來源；不得把未驗證的新聞推測寫成事實。
Qwen3 摘要若失敗，會自動使用程式化的保守摘要，不會讓 Email 整封失敗。
"""
from __future__ import annotations

import html
import json
import os
import re
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any

import requests


TAIPEI_TZ = "Asia/Taipei"
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_RECIPIENT = "a1113359@mail.nuk.edu.tw"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
def load_local_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

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


class EmailAgent:
    """盤後研究 Email Agent：先摘要，再寄送。"""

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir).resolve()
        self.env_path = self.base / ".env"
        self.env = load_local_env(self.env_path)
        self.output_dir = self.base / "output" / "research_reports"
        self.email_dir = self.base / "data" / "email"
        self.email_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _symbol_label(self, symbol: Any, report: dict[str, Any]) -> str:
        """股票顯示名稱：公司名稱（代號）。優先使用報告中的公司主檔，其次使用市場資料。"""
        code = str(symbol or "").strip().upper().replace(".TW", "")
        if not code:
            return ""

        profiles = report.get("business_profiles", {}) or {}
        profile = profiles.get(code, {}) if isinstance(profiles, dict) else {}
        name = str(profile.get("name", "") or "").strip()
        if name and name.lower() != "nan":
            return f"{name}（{code}）"

        for row in (report.get("financial_snapshots", []) or []):
            if isinstance(row, dict) and str(row.get("symbol", "")).strip().upper().replace(".TW", "") == code:
                name = str(row.get("name", "") or "").strip()
                if name and name.lower() != "nan":
                    return f"{name}（{code}）"

        mm = report.get("market_movers", {}) or {}
        for row in (mm.get("movers", []) or []):
            if isinstance(row, dict) and str(row.get("symbol", "")).strip().upper().replace(".TW", "") == code:
                name = str(row.get("name", "") or "").strip()
                if name and name.lower() != "nan":
                    return f"{name}（{code}）"

        return code

    # =======================================================================
    # Config / status
    # =======================================================================
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
        )

    def status(self) -> dict[str, Any]:
        c = self.config
        missing: list[str] = []

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
            "ollama_host": _cfg_value(self.env, "OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            "ollama_model": _cfg_value(self.env, "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            "env_path": str(self.env_path),
        }

    # =======================================================================
    # General helpers
    # =======================================================================
    @staticmethod
    def _esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        try:
            x = float(value)
            if x != x:  # NaN
                return default
            return x
        except Exception:
            return default

    @staticmethod
    def _pct(value: Any, assume_percent: bool = True) -> str:
        try:
            x = float(value)
            return f"{x:+.2f}%" if assume_percent else f"{x * 100:+.2f}%"
        except Exception:
            return "—"

    @staticmethod
    def _clean_llm_text(text: str) -> str:
        """移除 qwen3 可能輸出的 thinking 區塊，避免把思考過程寄給使用者。"""
        if not text:
            return ""

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None

        cleaned = EmailAgent._clean_llm_text(text)

        # ```json ... ```
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        candidates = []
        if fenced:
            candidates.append(fenced.group(1))
        candidates.append(cleaned)

        # 找第一個最外層 JSON 物件
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])

        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
            except Exception:
                continue

        return None

    def _table(self, headers: list[str], rows: list[list[Any]], row_classes: list[str] | None = None) -> str:
        if not rows:
            return "<p class='muted'>目前沒有可用資料。</p>"

        head = "".join(f"<th>{self._esc(h)}</th>" for h in headers)
        body: list[str] = []

        for idx, row in enumerate(rows):
            cls = row_classes[idx] if row_classes and idx < len(row_classes) else ""
            cells = "".join(f"<td>{self._esc(v)}</td>" for v in row)
            body.append(f"<tr class='{cls}'>{cells}</tr>")

        return (
            "<div class='table-wrap'><table><thead><tr>"
            + head
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )

    # =======================================================================
    # Build compact evidence for Qwen3
    # =======================================================================
    def _compact_evidence(self, report: dict[str, Any]) -> str:
        lines: list[str] = []

        report_date = str(report.get("report_date", ""))
        lines.append(f"研究日期：{report_date}")
        lines.append(f"研究結論：{report.get('research_conclusion', '')}")
        lines.append(f"市場狀態：{report.get('market_status', '')}")

        # 市場強弱
        mm = report.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
        movers = sorted(
            movers,
            key=lambda x: abs(self._safe_float(x.get("today_change_percent"), 0) or 0),
            reverse=True,
        )[:8]
        if movers:
            lines.append("市場波動候選：")
            for x in movers:
                lines.append(
                    f"- {x.get('name', x.get('symbol', ''))}({x.get('symbol', '')}) "
                    f"今日 {self._safe_float(x.get('today_change_percent'), 0):+.2f}%；"
                    f"5日 {self._safe_float(x.get('ret_5d'), 0):+.2f}%；"
                    f"產業 {x.get('industry_name', '')}；"
                    f"原因候選 {x.get('reason', '待查證')}"
                )

        industries = [x for x in mm.get("industry_summary", []) if isinstance(x, dict)]
        industries = sorted(
            industries,
            key=lambda x: self._safe_float(x.get("today_avg_change"), 0) or 0,
            reverse=True,
        )[:8]
        if industries:
            lines.append("產業強弱：")
            for x in industries:
                lines.append(
                    f"- {x.get('industry_name', '')} 今日平均 "
                    f"{self._safe_float(x.get('today_avg_change'), 0):+.2f}%；"
                    f"5日平均 {self._safe_float(x.get('ret_5d_avg'), 0):+.2f}%；"
                    f"漲跌 {x.get('rising_count', 0)}/{x.get('falling_count', 0)}"
                )

        # 法說會：只傳摘要欄位，不傳整份原文
        calls = [
            x for x in report.get("earnings_calls", [])
            if isinstance(x, dict) and not x.get("error")
        ]
        calls = sorted(calls, key=lambda x: str(x.get("event_date", "")), reverse=True)[:8]
        if calls:
            lines.append("最近法說會：")
            for x in calls:
                lines.append(
                    f"- {x.get('event_date', '')} {x.get('name', x.get('symbol', ''))}({x.get('symbol', '')})；"
                    f"訊號 {x.get('impact', '待確認')}；信心 {self._safe_float(x.get('confidence'), 0):.0f}；"
                    f"摘要 {x.get('one_line_summary') or x.get('summary') or '無摘要'}；"
                    f"財務 {'；'.join((x.get('financial_highlights') or [])[:3])}；"
                    f"展望 {'；'.join((x.get('guidance') or [])[:3])}；"
                    f"風險 {'；'.join((x.get('negative_factors') or [])[:3])}"
                )

        # 題材
        themes = [x for x in report.get("bullish_themes", []) if isinstance(x, dict)]
        themes = sorted(
            themes,
            key=lambda x: self._safe_float(x.get("strength"), 0) or 0,
            reverse=True,
        )[:6]
        if themes:
            lines.append("題材：")
            for x in themes:
                lines.append(
                    f"- {x.get('theme', '')} 強度 {self._safe_float(x.get('strength'), 0):.1f}/100；"
                    f"狀態 {x.get('status', '')}；"
                    f"官方 {x.get('official_evidence_count', 0)}；"
                    f"接單/出貨 {x.get('order_evidence_count', 0)}；"
                    f"營收支撐 {x.get('revenue_support_count', 0)}；"
                    f"查證 {('; '.join(x.get('contradiction_flags') or [])) or '無'}"
                )

        # 財報 / 營收
        financials = [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)]
        financials = sorted(
            financials,
            key=lambda x: self._safe_float(x.get("revenue_yoy"), -999) or -999,
            reverse=True,
        )[:10]
        if financials:
            lines.append("財報營收：")
            for x in financials:
                lines.append(
                    f"- {x.get('symbol', '')} 最新月營收 YoY {self._safe_float(x.get('revenue_yoy'), 0):+.2f}%；"
                    f"營業利益 {x.get('operating_profit', '')}；淨利 {x.get('net_income', '')}；"
                    f"EPS {x.get('eps', '')}"
                )

        return "\n".join(lines)

    # =======================================================================
    # Qwen3 Agent digest
    # =======================================================================
    def _fallback_digest(self, report: dict[str, Any]) -> dict[str, Any]:
        """Qwen3 無法使用時的保守摘要；只從結構化資料抽重點。"""
        mm = report.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]
        ups = sorted(
            [x for x in movers if (self._safe_float(x.get("today_change_percent"), 0) or 0) > 0],
            key=lambda x: self._safe_float(x.get("today_change_percent"), 0) or 0,
            reverse=True,
        )[:3]
        downs = sorted(
            [x for x in movers if (self._safe_float(x.get("today_change_percent"), 0) or 0) < 0],
            key=lambda x: self._safe_float(x.get("today_change_percent"), 0) or 0,
        )[:3]

        themes = sorted(
            [x for x in report.get("bullish_themes", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("strength"), 0) or 0,
            reverse=True,
        )[:3]

        calls = [
            x for x in report.get("earnings_calls", [])
            if isinstance(x, dict) and not x.get("error")
        ]
        calls = sorted(calls, key=lambda x: str(x.get("event_date", "")), reverse=True)[:5]

        financials = sorted(
            [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("revenue_yoy"), -999) or -999,
            reverse=True,
        )[:5]

        positives = [
            f"{x.get('name', x.get('symbol', ''))}({x.get('symbol', '')}) 今日 {self._safe_float(x.get('today_change_percent'), 0):+.2f}%"
            for x in ups
        ]
        positives += [
            f"題材：{x.get('theme', '')}（強度 {self._safe_float(x.get('strength'), 0):.1f}/100）"
            for x in themes
        ]

        negatives = [
            f"{x.get('name', x.get('symbol', ''))}({x.get('symbol', '')}) 今日 {self._safe_float(x.get('today_change_percent'), 0):+.2f}%"
            for x in downs
        ]

        watch = []
        if report.get("research_conclusion"):
            watch.append(str(report.get("research_conclusion")))
        for x in calls[:3]:
            if x.get("negative_factors"):
                watch.append(f"{x.get('symbol', '')} 法說風險：{'; '.join((x.get('negative_factors') or [])[:2])}")
        for x in themes:
            if x.get("contradiction_flags"):
                watch.append(f"{x.get('theme', '')}：" + "; ".join((x.get("contradiction_flags") or [])[:2]))

        earnings_summary = "目前沒有可用法說會資料。"
        if calls:
            earnings_summary = "；".join(
                f"{x.get('symbol', '')}：{(x.get('one_line_summary') or x.get('summary') or '')[:100]}"
                for x in calls[:3]
            )

        financial_focus = [
            f"{x.get('symbol', '')}：營收 YoY {self._safe_float(x.get('revenue_yoy'), 0):+.2f}%、EPS {x.get('eps', '—')}"
            for x in financials
        ]

        return {
            "overview": str(report.get("research_conclusion", "今日盤後研究已完成。")),
            "key_takeaways": (positives[:3] + negatives[:2])[:5],
            "positives": positives[:5],
            "negatives": negatives[:5],
            "watchlist": watch[:5] or ["請查看附件中的完整查證狀態。"],
            "earnings_summary": earnings_summary,
            "financial_focus": financial_focus[:5],
            "model_alert": str(report.get("research_conclusion", "")),
            "agent_status": "fallback",
        }

    def _agent_digest(self, report: dict[str, Any]) -> dict[str, Any]:
        """
        Email 不再重新決定研究重點。

        研究重點必須由 ResearchOrchestratorAgent 產生；EmailAgent
        只負責把 Agent 的研究筆記、標籤、證據與後續追蹤事項排版。
        若本次沒有 Agent 結果，才使用保守的程式化 fallback。
        """
        agent = report.get("agent_research_notes") or {}
        if isinstance(agent, dict) and (agent.get("overview") or agent.get("research_notes")):
            return {
                "overview": str(agent.get("overview", "")).strip(),
                "key_takeaways": self._as_list(agent.get("key_takeaways")),
                "positives": self._as_list(agent.get("positive_signals")),
                "negatives": self._as_list(agent.get("negative_signals")),
                "watchlist": self._as_list(agent.get("watch_items")),
                "earnings_summary": str(agent.get("earnings_digest", "")).strip(),
                "financial_focus": self._as_list(agent.get("financial_focus")),
                "cnyes_news_digest": self._as_list(agent.get("cnyes_news_digest")),
                "model_alert": "",
                "research_notes": agent.get("research_notes", []) if isinstance(agent.get("research_notes"), list) else [],
                "agent_actions": agent.get("agent_actions", []) if isinstance(agent.get("agent_actions"), list) else [],
                "agent_status": str(agent.get("agent_status", "qwen3_tool_agent")),
            }
        return self._fallback_digest(report)

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return [str(value).strip()]

    # =======================================================================
    # Executive HTML
    # =======================================================================
    def _digest_html(self, digest: dict[str, Any], report: dict[str, Any]) -> str:
        """Email 最上方的 Agent 研究摘要；不放法說會重複訊號、不放模型健康度。"""
        def bullet_list(items: list[str], css_class: str = "") -> str:
            if not items:
                return "<li>目前沒有可用資料。</li>"
            return "".join(
                f"<li class='{css_class}'>{self._esc(item)}</li>"
                for item in items[:10]
            )

        notes = digest.get("research_notes") or []
        importance_order = {"high": 0, "medium": 1, "low": 2}
        notes = sorted(
            [x for x in notes if isinstance(x, dict) and x.get("note")],
            key=lambda x: importance_order.get(str(x.get("importance", "medium")).lower(), 1),
        )

        note_blocks = []
        for n in notes[:30]:
            tags = "、".join(str(x) for x in (n.get("tags") or [])[:10]) or "—"
            evidence = "；".join(str(x) for x in (n.get("evidence") or [])[:8]) or "—"
            follow = "；".join(str(x) for x in (n.get("follow_up") or [])[:8]) or "—"
            links = [str(x).strip() for x in (n.get("source_links") or []) if str(x).strip()]
            conf = self._safe_float(n.get("confidence"), None)
            if conf is None:
                conf_text = "—"
            elif conf <= 1:
                conf_text = f"{conf * 100:.0f}%"
            else:
                conf_text = f"{conf:.0f}%"

            symbol = str(n.get("symbol", "")).strip()
            symbol_text = self._symbol_label(symbol, report) if symbol else ""
            meta_symbol = f"｜{symbol_text}" if symbol_text else ""
            source_html = ""
            if links:
                source_html = "<p><strong>🔗 來源：</strong>" + "　".join(
                    f"<a href='{self._esc(u)}'>來源 {i}</a>" for i, u in enumerate(links[:5], 1)
                ) + "</p>"

            note_blocks.append(
                f"<div class='research-note'>"
                f"<div><strong>{self._esc(n.get('title','研究發現'))}</strong> "
                f"<span class='note-meta'>{self._esc(n.get('importance','medium'))}｜{self._esc(n.get('type','other'))}"
                f"{meta_symbol}｜信心 {conf_text}</span></div>"
                f"<p>{self._esc(n.get('note',''))}</p>"
                f"<p><strong>🏷 記號：</strong>{self._esc(tags)}</p>"
                f"<p><strong>🔎 證據：</strong>{self._esc(evidence)}</p>"
                f"<p><strong>↪ 後續：</strong>{self._esc(follow)}</p>"
                f"{source_html}</div>"
            )

        status = self._esc(
            "Qwen3 自主工具研究 Agent"
            if digest.get("agent_status") == "qwen3_tool_agent"
            else ("Qwen3 Agent" if digest.get("agent_status") == "qwen3" else "保守 fallback")
        )
        overview = self._esc(digest.get("overview", "今日盤後研究已完成。"))

        # ---------------------------------------------------------------
        # ③ 目前主要利多題材：沿用 ResearchAgent 的題材證據，不讓 EmailAgent
        # 自己重新判斷，只負責呈現；公司名稱統一用「公司名稱（代號）」。
        # ---------------------------------------------------------------
        themes = sorted(
            [x for x in report.get("bullish_themes", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("strength"), 0) or 0,
            reverse=True,
        )[:3]

        theme_items: list[str] = []
        for t in themes:
            strength = self._safe_float(t.get("strength"), 0) or 0
            status_text = str(t.get("status", "") or "").strip()
            status_html = f"｜{self._esc(status_text)}" if status_text else ""
            theme_items.append(
                f"<li><strong>{self._esc(t.get('theme', ''))}</strong>｜"
                f"<span class='theme-score'>{strength:.1f}/100</span>{status_html}</li>"
            )
        theme_html = "".join(theme_items) or "<li>目前沒有通過證據門檻的主要利多題材。</li>"

        # ---------------------------------------------------------------
        # ④ 營收成長重點：最新月營收 YoY 由高到低，補上公司名稱。
        # ---------------------------------------------------------------
        financials = sorted(
            [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("revenue_yoy"), -999) or -999,
            reverse=True,
        )[:10]

        financial_items: list[str] = []
        for x in financials:
            symbol = x.get("symbol", "")
            label = self._symbol_label(symbol, report)
            yoy = self._safe_float(x.get("revenue_yoy"), None)
            eps = self._safe_float(x.get("eps"), None)
            yoy_text = f"{yoy:+.2f}%" if yoy is not None else "—"
            eps_text = f"{eps:.2f}" if eps is not None else "—"
            financial_items.append(
                f"<li><strong>{self._esc(label)}</strong>｜最新月營收 YoY "
                f"<span class='theme-score'>{self._esc(yoy_text)}</span>｜EPS {self._esc(eps_text)}</li>"
            )

        financial_html = "".join(financial_items)
        if not financial_html:
            extra = self._as_list(digest.get("financial_focus"))
            financial_html = (
                "".join(f"<li>{self._esc(x)}</li>" for x in extra[:10])
                if extra
                else "<li>Agent 已查詢，但目前沒有可驗證的財報／營收重點。</li>"
            )

        cnyes_digest = self._as_list(digest.get("cnyes_news_digest"))
        cnyes_research = report.get("cnyes_research_digest", {}) or {}
        cnyes_findings = cnyes_research.get("key_findings", []) if isinstance(cnyes_research, dict) else []
        cnyes_blocks = []
        for item in cnyes_findings[:8]:
            if not isinstance(item, dict):
                continue
            conf = self._safe_float(item.get("confidence"), 0) or 0
            conf_text = f"{conf * 100:.0f}%" if conf <= 1 else f"{conf:.0f}%"
            links = [str(x).strip() for x in (item.get("source_links") or []) if str(x).strip()]
            source_html = ""
            if links:
                source_html = "<p class='muted'>來源：" + "　".join(
                    f"<a href='{self._esc(u)}'>鉅亨新聞{i}</a>" for i, u in enumerate(links[:3], 1)
                ) + "</p>"
            symbols = "、".join(str(x) for x in (item.get("related_symbols") or [])[:10])
            symbol_html = f"｜關聯：{self._esc(symbols)}" if symbols else ""
            cnyes_blocks.append(
                f"<div class='research-note'>"
                f"<div><strong>{self._esc(item.get('title','新聞研究'))}</strong> "
                f"<span class='note-meta'>{self._esc(item.get('impact','待查證'))}｜信心 {conf_text}{symbol_html}</span></div>"
                f"<p><strong>發生：</strong>{self._esc(item.get('summary',''))}</p>"
                f"<p><strong>研究意義：</strong>{self._esc(item.get('why_relevant',''))}</p>"
                f"<p><strong>🔎 證據：</strong>{self._esc('；'.join(str(x) for x in (item.get('evidence') or [])[:5]) or '—')}</p>"
                f"{source_html}"
                f"</div>"
            )
        if not cnyes_blocks and cnyes_digest:
            cnyes_blocks = [f"<p>{self._esc(x)}</p>" for x in cnyes_digest[:8]]
        cache_dates = (report.get("cnyes_news", {}) or {}).get("cache_dates", [])
        cache_count = int((report.get("cnyes_news", {}) or {}).get("article_count", 0) or 0)
        cache_note = (
            f"已讀夜間新聞快取：{'、'.join(str(x) for x in cache_dates)}｜共 {cache_count} 篇；盤後研究未重新爬網站。"
            if cache_dates else "本次沒有可用的 CNYES 夜間新聞快取。"
        )

        return f"""
        <section class='agent-box'>
          <div class='agent-title'>🤖 AI 研究助理｜今日盤後研究 <span class='agent-badge'>{status}</span></div>
          <div class='overview'>{overview}</div>

          <div class='mini-card'>
            <h3>① 今日 AI 研究重點</h3>
            <ol>{bullet_list(self._as_list(digest.get("key_takeaways")))}</ol>
          </div>

          <div class='grid3'>
            <div class='signal positive-box'>
              <h3>正向訊號</h3>
              <ul>{bullet_list(self._as_list(digest.get("positives")), "positive")}</ul>
            </div>
            <div class='signal negative-box'>
              <h3>負向訊號</h3>
              <ul>{bullet_list(self._as_list(digest.get("negatives")), "negative")}</ul>
            </div>
            <div class='signal watch-box'>
              <h3>待追蹤／待查證</h3>
              <ul>{bullet_list(self._as_list(digest.get("watchlist")), "watch")}</ul>
            </div>
          </div>

          <div class='mini-card'>
            <h3>④ 目前主要利多題材</h3>
            <ul>{theme_html}</ul>
          </div>

          <div class='mini-card'>
            <h3>⑤ 營收成長重點</h3>
            <ul>{financial_html}</ul>
          </div>

          {("<div class='mini-card'><h3>Agent 研究筆記</h3>" + ''.join(note_blocks) + "</div>") if note_blocks else ""}
        </section>
        """

    # =======================================================================
    # Market section
    # =======================================================================
    def _market_section(self, report: dict[str, Any]) -> str:
        mm = report.get("market_movers", {}) or {}
        movers = [x for x in mm.get("movers", []) if isinstance(x, dict)]

        ups = sorted(
            [x for x in movers if (self._safe_float(x.get("today_change_percent"), 0) or 0) > 0],
            key=lambda x: self._safe_float(x.get("today_change_percent"), 0) or 0,
            reverse=True,
        )[:5]
        downs = sorted(
            [x for x in movers if (self._safe_float(x.get("today_change_percent"), 0) or 0) < 0],
            key=lambda x: self._safe_float(x.get("today_change_percent"), 0) or 0,
        )[:5]

        profiles = report.get("business_profiles", {}) or {}
        data_asof = str(mm.get("data_asof", "") or "")
        data_date = data_asof[:10] if len(data_asof) >= 10 else ""

        def industry_business(x: dict[str, Any]) -> str:
            symbol = str(x.get("symbol", "")).strip().upper().replace(".TW", "")
            p = profiles.get(symbol, {}) if isinstance(profiles, dict) else {}
            business = str(
                p.get("business_group")
                or p.get("primary_chain")
                or x.get("business_group")
                or x.get("primary_chain")
                or p.get("industry_name")
                or x.get("industry_name")
                or ""
            ).strip()
            return business or "未分類"

        up_rows = []
        for x in ups:
            up_rows.append([
                f"{x.get('name', x.get('symbol', ''))} ({x.get('symbol', '')})",
                industry_business(x),
                f"{self._safe_float(x.get('today_change_percent'), 0):+.2f}%",
                f"{self._safe_float(x.get('ret_5d'), 0):+.2f}%" if x.get("ret_5d") is not None else "—",
            ])

        down_rows = []
        for x in downs:
            down_rows.append([
                f"{x.get('name', x.get('symbol', ''))} ({x.get('symbol', '')})",
                industry_business(x),
                f"{self._safe_float(x.get('today_change_percent'), 0):+.2f}%",
                f"{self._safe_float(x.get('ret_5d'), 0):+.2f}%" if x.get("ret_5d") is not None else "—",
            ])

        industries = [x for x in mm.get("industry_summary", []) if isinstance(x, dict)]
        industries = sorted(
            industries,
            key=lambda x: self._safe_float(x.get("today_avg_change"), 0) or 0,
            reverse=True,
        )[:6]
        industry_rows = [
            [
                f"{x.get('industry_code', '')} {x.get('industry_name', '')}",
                x.get("candidate_count", ""),
                f"{self._safe_float(x.get('today_avg_change'), 0):+.2f}%",
                f"{self._safe_float(x.get('ret_5d_avg'), 0):+.2f}%",
                f"{x.get('rising_count', 0)} / {x.get('falling_count', 0)}",
            ]
            for x in industries
        ]

        freshness_note = ""
        if data_date:
            if data_date == str(report.get("report_date", "")):
                freshness_note = f"<p class='muted'>行情資料日：{self._esc(data_date)}（當日資料）</p>"
            else:
                freshness_note = f"<p class='notice'>行情資料日為 {self._esc(data_date)}，與研究日期 {self._esc(report.get('report_date',''))} 不一致；為避免誤報舊資料，本次不應視為今日行情。</p>"

        return f"""
        <section>
          <h2>一、今日市場重點</h2>
          {freshness_note}
          <div class='legend'><span class='positive-text'>紅字＝上漲／利多</span>　<span class='negative-text'>綠字＝下跌／利空</span>　<span class='watch-text'>橘字＝待追蹤</span></div>
          <h3>漲幅重點</h3>
          {self._colored_table(["公司", "產業／業務", "今日", "5日"], up_rows, "positive")}
          <h3>跌幅重點</h3>
          {self._colored_table(["公司", "產業／業務", "今日", "5日"], down_rows, "negative")}
          <h3>產業強弱 Top 6</h3>
          {self._table(["產業", "樣本", "今日平均", "5日平均", "漲/跌"], industry_rows)}
        </section>
        """

    def _colored_table(self, headers: list[str], rows: list[list[Any]], signal: str) -> str:
        if not rows:
            return "<p class='muted'>目前沒有可用資料。</p>"

        head = "".join(f"<th>{self._esc(h)}</th>" for h in headers)
        body = []
        for row in rows:
            cells: list[str] = []
            for idx, value in enumerate(row):
                if idx == 2:
                    cls = "positive-text" if signal == "positive" else "negative-text"
                    cells.append(f"<td class='{cls}'><strong>{self._esc(value)}</strong></td>")
                else:
                    cells.append(f"<td>{self._esc(value)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")

        return (
            "<div class='table-wrap'><table><thead><tr>"
            + head
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )

    # =======================================================================
    # Earnings section
    # =======================================================================
    def _earnings_section(self, report: dict[str, Any]) -> str:
        calls = [
            x for x in report.get("earnings_calls", [])
            if isinstance(x, dict) and not x.get("error")
        ]
        calls = sorted(calls, key=lambda x: str(x.get("event_date", "")), reverse=True)[:10]

        if not calls:
            return """
            <section>
              <h2>二、最近 5 天法說會</h2>
              <div class='notice'>目前沒有成功擷取到可用的 Fugle 法說會摘要。</div>
            </section>
            """

        blocks: list[str] = []
        for x in calls:
            verified = bool(x.get("detail_read_verified") or x.get("memo_opened"))
            impact = str(x.get("impact", "待確認"))
            if "利多" in impact or "正" in impact:
                signal_class = "positive-text"
            elif "利空" in impact or "負" in impact:
                signal_class = "negative-text"
            else:
                signal_class = "watch-text"

            summary = x.get("one_line_summary") or x.get("summary") or "目前沒有 AI 摘要。"
            financial = x.get("financial_highlights") or []
            guidance = x.get("guidance") or []
            positive = x.get("positive_factors") or []
            negative = x.get("negative_factors") or []
            qa = x.get("qa_highlights") or []
            source = x.get("source_url") or x.get("url") or ""

            verification = (
                "Fugle 詳細備忘錄＋Ollama Agent"
                if verified
                else "尚未完成詳細正文驗證"
            )

            reaction = x.get("price_reaction_label") or "尚無法計算"
            post1 = self._pct(x.get("post_1d_return")) if x.get("post_1d_return") is not None else "—"
            post5 = self._pct(x.get("post_5d_return")) if x.get("post_5d_return") is not None else "—"

            def lis(items: list[Any]) -> str:
                clean = [str(i).strip() for i in items if str(i).strip()]
                return "".join(f"<li>{self._esc(i)}</li>" for i in clean[:5]) or "<li>無資料</li>"

            source_html = (
                f"<p><a href='{self._esc(source)}'>查看 Fugle 原始法說會備忘錄</a></p>"
                if source
                else ""
            )

            blocks.append(
                f"""
                <details class='earnings-card'>
                  <summary>
                    <strong>{self._esc(x.get('event_date',''))}｜{self._esc(x.get('name', x.get('symbol','')))}({self._esc(x.get('symbol',''))})</strong>
                    <span class='{signal_class}'>　{self._esc(impact)}</span>
                    <span class='muted'>　信心 {self._safe_float(x.get('confidence'),0):.0f}%</span>
                  </summary>
                  <div class='details-body'>
                    <p class='summary-line'><strong>一句話摘要：</strong>{self._esc(summary)}</p>
                    <p><strong>證據：</strong>{self._esc(verification)}</p>
                    <div class='grid2'>
                      <div><h4>財務重點</h4><ul>{lis(financial)}</ul></div>
                      <div><h4>展望／指引</h4><ul>{lis(guidance)}</ul></div>
                    </div>
                    <div class='grid2'>
                      <div><h4 class='positive-text'>利多因素</h4><ul>{lis(positive)}</ul></div>
                      <div><h4 class='negative-text'>利空／風險</h4><ul>{lis(negative)}</ul></div>
                    </div>
                    {f"<h4>Q&A 重點</h4><ul>{lis(qa)}</ul>" if qa else ""}
                    <p><strong>法說後反應：</strong>{self._esc(reaction)}｜後1日 {post1}｜後5日 {post5}</p>
                    {source_html}
                  </div>
                </details>
                """
            )

        return """
        <section>
          <h2>二、最近 5 天法說會</h2>
          <p class='muted'>只顯示最近 5 天的法說事件。摘要由 Fugle 詳細備忘錄經 Ollama Agent 整理；原始內容不直接塞進 Email 本文。</p>
          <div class='earnings-list'>%s</div>
        </section>
        """ % "".join(blocks)

    # =======================================================================
    # Themes
    # =======================================================================
    def _themes_section(self, report: dict[str, Any]) -> str:
        themes = [x for x in report.get("bullish_themes", []) if isinstance(x, dict)]
        themes = sorted(
            themes,
            key=lambda x: self._safe_float(x.get("strength"), 0) or 0,
            reverse=True,
        )[:3]

        if not themes:
            return """
            <section>
              <h2>④ 目前主要利多題材</h2>
              <div class='notice'>Agent 已檢查目前題材資料，但本次沒有形成足夠證據的主要利多題材。</div>
            </section>
            """

        rows = []
        for x in themes:
            rows.append([
                x.get("theme", ""),
                f"{self._safe_float(x.get('strength'), 0):.1f}/100",
                x.get("status", ""),
            ])

        # Agent 研究筆記若提到同一題材，將原因摘要放在表格下方，避免只看到分數。
        agent = report.get("agent_research_notes") or {}
        notes = agent.get("research_notes", []) if isinstance(agent, dict) else []
        theme_names = [str(x.get("theme", "")).strip() for x in themes]
        reason_rows = []
        for note in notes:
            if not isinstance(note, dict) or not note.get("note"):
                continue
            blob = f"{note.get('title','')} {note.get('note','')} {' '.join(map(str, note.get('tags') or []))}"
            for t in theme_names:
                if t and t in blob:
                    reason_rows.append(f"{note.get('title','研究發現')}：{note.get('note','')}")
                    break
            if len(reason_rows) >= 3:
                break

        reasons_html = (
            "<div class='mini-card'><h3>Agent 對題材的補充研究</h3><ul>"
            + "".join(f"<li>{self._esc(x)}</li>" for x in reason_rows)
            + "</ul></div>"
            if reason_rows else ""
        )

        return f"""
        <section>
          <h2>④ 目前主要利多題材</h2>
          <p class='muted'>以下為研究引擎依新聞、產業鏈、接單／出貨與營收支撐計算出的主要題材；Agent 另外負責查找近期市場原因與補充證據。</p>
          {self._table(["題材", "強度", "狀態"], rows)}
          {reasons_html}
        </section>
        """

    # =======================================================================
    # Financials
    # =======================================================================
    def _financial_section(self, report: dict[str, Any]) -> str:
        rows_data = [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)]
        rows_data = sorted(
            rows_data,
            key=lambda x: self._safe_float(x.get("revenue_yoy"), -999) or -999,
            reverse=True,
        )[:10]

        rows = []
        for x in rows_data:
            yoy = self._safe_float(x.get("revenue_yoy"), 0) or 0
            name_label = self._symbol_label(x.get("symbol", ""), report)
            rows.append([
                name_label,
                x.get("latest_month_revenue", ""),
                f"{yoy:+.2f}%",
                x.get("eps", ""),
            ])

        table = self._table(["公司", "最新月營收", "YoY", "EPS"], rows)
        if rows:
            head = "<tr><th>公司</th><th>最新月營收</th><th>YoY</th><th>EPS</th></tr>"
            body = []
            for row in rows:
                yoy = str(row[2])
                cls = "positive-text" if yoy.startswith("+") and not yoy.startswith("+0.00") else "negative-text" if yoy.startswith("-") else ""
                body.append(
                    f"<tr><td>{self._esc(row[0])}</td><td>{self._esc(row[1])}</td><td class='{cls}'><strong>{self._esc(yoy)}</strong></td><td>{self._esc(row[3])}</td></tr>"
                )
            table = "<div class='table-wrap'><table><thead>" + head + "</thead><tbody>" + "".join(body) + "</tbody></table></div>"

        # 若結構化財報資料為空，Agent 不應只回報「沒有資料」；
        # 這裡顯示 Agent 的補充研究結果，或明確告知 Agent 已查詢仍無法驗證。
        agent = report.get("agent_research_notes") or {}
        financial_focus = self._as_list(agent.get("financial_focus")) if isinstance(agent, dict) else []
        actions = agent.get("agent_actions", []) if isinstance(agent, dict) else []
        queried_financial = any(
            isinstance(x, dict) and str(x.get("tool", "")) in {"get_company_financial", "search_news_evidence", "search_live_news"}
            for x in actions
        )

        if not rows and financial_focus:
            table = "<div class='mini-card'><h3>Agent 財報／營收補充研究</h3><ul>" + "".join(
                f"<li>{self._esc(x)}</li>" for x in financial_focus[:10]
            ) + "</ul></div>"
        elif not rows and queried_financial:
            table = "<div class='notice'>Agent 已主動查詢財報／營收與相關新聞，但本次仍沒有找到可驗證的結構化數據。</div>"
        elif not rows:
            table = "<div class='notice'>本次沒有可用的財報／營收數據。正常研究流程會由 Agent 進一步查詢新聞與可用公開資料。</div>"

        return f"""
        <section>
          <h2>⑤ 營收成長重點</h2>
          <p class='muted'>正文列出目前最值得注意的營收 YoY / EPS 變化；完整財報數據保留於附件。若結構化資料不足，Agent 會先主動補查新聞與公開資料。</p>
          {table}
        </section>
        """

    # =======================================================================
    # Model health
    # =======================================================================
    def _model_section(self, report: dict[str, Any]) -> str:
        """模型健康度保留於研究 JSON / 附件，不在 Email 正文顯示。"""
        return ""

    def _source_section(self, report: dict[str, Any]) -> str:
        sources = report.get("official_sources", {}) or {}
        source_text = (
            f"TWSE公告 {sources.get('twse_announcements_rows', 0)} 筆、"
            f"TPEx公告 {sources.get('tpex_announcements_rows', 0)} 筆、"
            f"TWSE營收 {sources.get('twse_revenue_rows', 0)} 筆、"
            f"TPEx營收 {sources.get('tpex_revenue_rows', 0)} 筆、"
            f"TWSE財報 {sources.get('twse_income_rows', 0)} 筆、"
            f"TPEx財報 {sources.get('tpex_income_rows', 0)} 筆、"
            f"鉅亨近兩日新聞 {sources.get('cnyes_news_articles', 0)} 篇"
        )
        return f"""
        <section>
          <h2>五、資料來源與查證</h2>
          <p>{self._esc(source_text)}</p>
          <p class='muted'>法說會、重大訊息與財報/營收以公開原始資料為優先；新聞與題材僅作研究線索。未完成官方交叉驗證者標示為待查證。</p>
          <p class='muted'>鉅亨新聞由前一晚 23:00 夜間爬蟲建立日檔；盤後讀取最近兩份快取，再由新聞 Agent 與主研究 Agent 整合，盤後不重新爬網站。新聞屬研究線索，重大事項仍需回看原始公告。</p>
          <p class='muted'>完整 research JSON / Markdown 已附加於本信；需要更完整證據時請查看附件與原始來源。</p>
        </section>
        """

    # =======================================================================
    # Full HTML / text
    # =======================================================================
    def build_html(self, report: dict[str, Any]) -> str:
        report_date = self._esc(report.get("report_date", ""))
        watchlist = report.get("research_scope", {}).get("watchlist_symbols", []) or []
        digest = self._agent_digest(report)

        return f"""<!doctype html>
<html lang='zh-Hant'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
body{{font-family:Arial,'Microsoft JhengHei','Noto Sans TC',sans-serif;background:#f3f6fa;color:#202124;line-height:1.65;margin:0;padding:0}}
.container{{max-width:1050px;margin:0 auto;background:#fff;padding:26px 30px}}
h1{{margin:0 0 4px;font-size:28px;color:#111827}}
h2{{margin-top:28px;border-bottom:2px solid #e5e7eb;padding-bottom:7px;font-size:20px}}
h3{{margin-top:16px;font-size:16px}}
h4{{margin-bottom:7px}}
p{{margin:8px 0}}
ul,ol{{margin-top:6px}}
.muted{{color:#6b7280;font-size:13px}}
.legend{{font-size:13px;margin:6px 0 12px}}
.positive-text{{color:#dc2626!important}}
.negative-text{{color:#16a34a!important}}
.watch-text{{color:#d97706!important}}
.positive-row{{background:#fff7f7}}
.negative-row{{background:#f4fff7}}
.table-wrap{{overflow-x:auto;margin:8px 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #e5e7eb;padding:7px 8px;vertical-align:top;text-align:left}}
th{{background:#f8fafc}}
.agent-box{{background:#f8fafc;border:1px solid #dbeafe;border-radius:12px;padding:18px;margin:20px 0}}
.agent-title{{font-size:20px;font-weight:700;margin-bottom:8px}}
.agent-badge{{font-size:11px;padding:3px 8px;border-radius:999px;background:#e0ecff;color:#2563eb;margin-left:6px}}
.overview{{font-size:17px;font-weight:700;background:#fff;border-left:4px solid #2563eb;padding:12px 14px;margin:12px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.mini-card{{background:#fff;border:1px solid #e5e7eb;border-radius:9px;padding:12px;margin-top:10px}}
.signal{{border-radius:9px;padding:12px;background:#fff;border:1px solid #e5e7eb}}
.positive-box{{background:#fff8f8;border-color:#fecaca}}
.negative-box{{background:#f7fff9;border-color:#bbf7d0}}
.watch-box{{background:#fffbf2;border-color:#fed7aa}}
.notice{{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px;margin:10px 0}}
.earnings-list{{margin-top:10px}}
.earnings-card{{border:1px solid #e5e7eb;border-radius:9px;background:#fff;margin:9px 0}}
.earnings-card summary{{cursor:pointer;padding:12px 14px;list-style:none}}
.earnings-card summary::-webkit-details-marker{{display:none}}
.details-body{{padding:0 14px 14px;border-top:1px solid #f0f0f0}}
.summary-line{{background:#f8fafc;border-left:4px solid #64748b;padding:10px}}
a{{color:#2563eb;text-decoration:none}}
a:hover{{text-decoration:underline}}
.footer{{margin-top:30px;padding-top:14px;border-top:1px solid #e5e7eb;color:#6b7280;font-size:12px}}
@media(max-width:760px){{.grid2,.grid3{{grid-template-columns:1fr}}.container{{padding:18px 14px}}}}
</style>
</head>
<body>
<div class='container'>
  <h1>AI 台股盤後財報</h1>
  <div class='muted'>研究日期：{report_date}｜本機 Qwen3 8B + Fugle / TWSE / TPEx / MOPS 證據整合</div>
  <p><strong>研究自選股：</strong>{self._esc('、'.join(map(str, watchlist)))}</p>

  {self._digest_html(digest, report)}
  {self._market_section(report)}
  {self._earnings_section(report)}
  {self._themes_section(report)}
  {self._financial_section(report)}
  {self._source_section(report)}

  <div class='footer'>
    本信的正文是 AI 研究摘要，不重複貼上完整研究資料。重大消息請回看 MOPS / TWSE / TPEx / Fugle 原始資料。<br>
    本報告為盤後研究與模型治理工具，不構成投資建議。
  </div>
</div>
</body>
</html>"""

    def build_text(self, report: dict[str, Any]) -> str:
        digest = self._agent_digest(report)
        lines = [
            f"AI 台股盤後財報｜{report.get('report_date', '')}",
            "",
            "【① 今日 AI 研究重點】",
            str(digest.get("overview", "今日盤後研究已完成。")),
            "",
        ]
        lines += [f"- {x}" for x in self._as_list(digest.get("key_takeaways"))[:8]] or ["- 目前沒有可用資料。"]

        lines += ["", "【Agent 研究筆記】"]
        for n in digest.get("research_notes", [])[:30]:
            if not isinstance(n, dict):
                continue
            conf = self._safe_float(n.get("confidence"), None)
            conf_text = f"{conf * 100:.0f}%" if conf is not None and conf <= 1 else (f"{conf:.0f}%" if conf is not None else "—")
            symbol = self._symbol_label(n.get("symbol", ""), report) if n.get("symbol") else ""
            suffix = f"｜{symbol}" if symbol else ""
            lines += [
                f"- {n.get('title','研究發現')} {n.get('importance','medium')}｜{n.get('type','other')}{suffix}｜信心 {conf_text}",
                f"  {n.get('note','')}",
                f"  記號：{'、'.join(map(str, n.get('tags') or [])) or '—'}",
                f"  證據：{'；'.join(map(str, n.get('evidence') or [])) or '—'}",
                f"  後續：{'；'.join(map(str, n.get('follow_up') or [])) or '—'}",
            ]

        cnyes_digest = self._as_list(digest.get("cnyes_news_digest"))
        if cnyes_digest:
            lines += ["", "鉅亨近兩日新聞｜Agent 研究摘要"]
            lines += [f"- {x}" for x in cnyes_digest[:15]]

        lines += ["", "④ 目前主要利多題材"]
        themes = sorted(
            [x for x in report.get("bullish_themes", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("strength"), 0) or 0,
            reverse=True,
        )[:3]
        lines += [
            f"- {x.get('theme','')}｜{self._safe_float(x.get('strength'),0):.1f}/100｜{x.get('status','')}"
            for x in themes
        ] or ["- 本次沒有足夠證據的主要利多題材。"]

        lines += ["", "⑤ 營收成長重點"]
        financials = sorted(
            [x for x in report.get("financial_snapshots", []) if isinstance(x, dict)],
            key=lambda x: self._safe_float(x.get("revenue_yoy"), -999) or -999,
            reverse=True,
        )[:10]
        if financials:
            lines += [
                f"- {self._symbol_label(x.get('symbol',''), report)}｜最新月營收 {x.get('latest_month_revenue','')}｜YoY {self._safe_float(x.get('revenue_yoy'),0):+.2f}%｜EPS {x.get('eps','—')}"
                for x in financials
            ]
        else:
            financial_focus = self._as_list(digest.get("financial_focus"))
            lines += [f"- {x}" for x in financial_focus[:10]] if financial_focus else ["- Agent 已查詢但目前仍無可驗證的財報／營收資料。"]

        lines += ["", "最近 5 天法說會請見 Email 的②區塊。", "完整研究 JSON / Markdown 已附檔。"]
        return "\n".join(lines)

    # =======================================================================
    # Morning report
    # =======================================================================
    def _morning_news_section(self, report: dict[str, Any]) -> str:
        morning = report.get("morning_agent", {}) or {}
        news = [x for x in (morning.get("news_summary", []) or []) if isinstance(x, dict)]
        if not news:
            return "<div class='notice'>夜間 CNYES 新聞快取已讀取，但早報 Agent 本次沒有形成可驗證的新聞摘要。</div>"
        blocks = []
        for x in news[:10]:
            conf = self._safe_float(x.get("confidence"), None)
            conf_text = "—" if conf is None else (f"{conf*100:.0f}%" if conf <= 1 else f"{conf:.0f}%")
            links = [str(v).strip() for v in (x.get("source_links") or []) if str(v).strip()]
            source_html = "".join(f"<a href='{self._esc(u)}'>來源{i}</a>　" for i, u in enumerate(links[:3], 1))
            blocks.append(
                f"<div class='research-note'>"
                f"<div><strong>{self._esc(x.get('title','財經事件'))}</strong> "
                f"<span class='note-meta'>{self._esc(x.get('impact','待查證'))}｜信心 {conf_text}</span></div>"
                f"<p>{self._esc(x.get('summary',''))}</p>"
                f"<p><strong>影響產業：</strong>{self._esc('、'.join(map(str, x.get('industries') or [])) or '—')}</p>"
                f"<p><strong>🔎 證據：</strong>{self._esc('；'.join(map(str, x.get('evidence') or [])) or '—')}</p>"
                f"{('<p>'+source_html+'</p>') if source_html else ''}</div>"
            )
        return "<div class='earnings-list'>" + "".join(blocks) + "</div>"

    def build_morning_html(self, report: dict[str, Any]) -> str:
        report_date = self._esc(report.get("report_date", ""))
        watchlist = report.get("symbols", []) or report.get("research_scope", {}).get("watchlist_symbols", []) or []
        morning = report.get("morning_agent", {}) or {}
        industries = [x for x in (morning.get("recommended_industries", []) or []) if isinstance(x, dict)]
        stocks = [x for x in (morning.get("recommended_stocks", []) or []) if isinstance(x, dict)]
        watch = [x for x in (morning.get("watch_items", []) or []) if str(x).strip()]
        industry_html = "".join(
            f"<li><strong>{self._esc(x.get('industry',''))}</strong>｜{self._esc(x.get('why',''))}"
            f"<br><span class='muted'>相關觀察：{self._esc('、'.join(map(str, x.get('related_symbols') or [])) or '—')}</span></li>"
            for x in industries[:10]
        ) or "<li>本次 Agent 沒有形成足夠證據的產業觀察。</li>"
        stock_blocks = []
        for x in stocks[:12]:
            links = [str(v).strip() for v in (x.get("source_links") or []) if str(v).strip()]
            source_html = "　".join(f"<a href='{self._esc(u)}'>來源{i}</a>" for i, u in enumerate(links[:3], 1))
            stock_blocks.append(
                f"<div class='research-note'><div><strong>{self._esc(x.get('name', x.get('symbol','')))}（{self._esc(x.get('symbol',''))}）</strong>"
                f"<span class='note-meta'>｜{self._esc(x.get('industry',''))}</span></div>"
                f"<p><strong>研究原因：</strong>{self._esc(x.get('reason',''))}</p>"
                f"<p><strong>證據：</strong>{self._esc('；'.join(map(str, x.get('evidence') or [])) or '—')}</p>"
                f"{('<p>'+source_html+'</p>') if source_html else ''}</div>"
            )
        watch_html = "<ul>" + "".join(f"<li>{self._esc(x)}</li>" for x in watch[:10]) + "</ul>" if watch else ""
        cache = report.get("cnyes_news", {}) or {}
        cache_note = f"夜間快取日期：{'、'.join(map(str, cache.get('cache_dates', []) or [])) or '—'}｜共 {int(cache.get('article_count', 0) or 0)} 篇"
        return f"""<!doctype html>
<html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<style>
body{{font-family:Arial,'Microsoft JhengHei','Noto Sans TC',sans-serif;background:#f3f6fa;color:#202124;line-height:1.65;margin:0;padding:0}}
.container{{max-width:1050px;margin:0 auto;background:#fff;padding:26px 30px}}
h1{{margin:0 0 4px;font-size:28px;color:#111827}} h2{{margin-top:28px;border-bottom:2px solid #e5e7eb;padding-bottom:7px;font-size:20px}} h3{{margin-top:16px;font-size:16px}}
.muted{{color:#6b7280;font-size:13px}} .research-note{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:13px 15px;margin:10px 0}}
.note-meta{{font-size:12px;color:#6b7280;margin-left:6px}} .notice{{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px;margin:10px 0}}
.positive-text{{color:#dc2626}} .negative-text{{color:#16a34a}} a{{color:#2563eb;text-decoration:none}} .footer{{margin-top:30px;padding-top:14px;border-top:1px solid #e5e7eb;color:#6b7280;font-size:12px}}
</style></head><body><div class='container'>
<h1>AI 台股早報</h1>
<div class='muted'>研究日期：{report_date}｜本機 Qwen3 8B｜Fugle 法說會＋鉅亨近兩日財經新聞</div>
<p><strong>研究自選股：</strong>{self._esc('、'.join(map(str, watchlist)))}</p>
<div class='research-note'><strong>🤖 AI 早報研究助理</strong><p>{self._esc(morning.get('overview','今天早報研究已完成。'))}</p><p class='muted'>{self._esc(cache_note)}</p></div>
{self._earnings_section(report).replace('二、最近 5 天法說會','1. 最近 5 天法說會')}
<section><h2>2. 最近 2 天財經報導摘要</h2>{self._morning_news_section(report)}</section>
<section><h2>3. Agent 看好的產業類別與股票</h2><h3>看好的產業類別</h3><ul>{industry_html}</ul><h3>看好的股票</h3>{''.join(stock_blocks) or '<div class="notice">本次沒有形成足夠證據的股票觀察名單。</div>'}</section>
{('<section><h2>需要持續追蹤</h2>'+watch_html+'</section>') if watch_html else ''}
<div class='footer'>新聞在前一晚 23:00 由夜間爬蟲完成；早報只讀本機快取。重大消息請回看原始公告。<br>本報告為研究工具，不構成投資建議。</div>
</div></body></html>"""

    def build_morning_text(self, report: dict[str, Any]) -> str:
        morning = report.get("morning_agent", {}) or {}
        lines = [f"AI 台股早報｜{report.get('report_date','')}", "", "【1. 最近 5 天法說會】"]
        calls = [x for x in (report.get("earnings_calls", []) or []) if isinstance(x, dict) and not x.get("error")]
        for x in calls[:20]:
            lines.append(f"- {x.get('event_date')} {x.get('name',x.get('symbol'))}({x.get('symbol')})｜{x.get('impact')}｜{x.get('one_line_summary') or x.get('summary','')}")
        if not calls: lines.append("- 最近 5 天沒有可用法說會資料。")
        lines += ["", "【2. 最近 2 天財經報導摘要】"]
        for x in (morning.get("news_summary", []) or [])[:10]:
            if isinstance(x, dict): lines.append(f"- {x.get('title','')}｜{x.get('impact','待查證')}｜{x.get('summary','')}")
        lines += ["", "【3. Agent 看好的產業類別與股票】", "", "看好的產業類別："]
        for x in (morning.get("recommended_industries", []) or [])[:10]:
            if isinstance(x, dict): lines.append(f"- {x.get('industry','')}｜{x.get('why','')}")
        lines += ["", "看好的股票："]
        for x in (morning.get("recommended_stocks", []) or [])[:12]:
            if isinstance(x, dict): lines.append(f"- {x.get('name',x.get('symbol'))}（{x.get('symbol')}）｜{x.get('industry','')}｜{x.get('reason','')}")
        return "\n".join(lines)

    # =======================================================================
    # SMTP
    # =======================================================================
    def _smtp_send_once(self, message: EmailMessage, host: str, port: int, use_ssl: bool) -> list[str]:
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
        c = self.config
        primary = (c.smtp_host, c.smtp_port, c.use_ssl or c.smtp_port == 465)
        fallback = (c.smtp_host, 587, False) if c.smtp_port == 465 else (c.smtp_host, 465, True)
        return [primary] if primary == fallback else [primary, fallback]

    def _smtp_send(self, message: EmailMessage) -> list[str]:
        errors: list[str] = []
        for host, port, use_ssl in self._smtp_candidates():
            try:
                return self._smtp_send_once(message, host, port, use_ssl)
            except smtplib.SMTPAuthenticationError:
                raise
            except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, smtplib.SMTPException, OSError) as exc:
                errors.append(f"{host}:{port} {type(exc).__name__}: {exc}")
        raise smtplib.SMTPServerDisconnected(" | ".join(errors) or "SMTP connection failed")

    def diagnose_smtp(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sender": self.config.sender,
            "smtp_host": self.config.smtp_host,
            "candidates": [],
        }

        status = self.status()
        if not status["configured"]:
            result["configured"] = False
            result["error"] = status["missing"]
            return result

        result["configured"] = True
        for host, port, use_ssl in self._smtp_candidates():
            item: dict[str, Any] = {
                "host": host,
                "port": port,
                "use_ssl": use_ssl,
                "ok": False,
                "steps": [],
            }

            try:
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
            except (smtplib.SMTPException, OSError) as exc:
                item["error_type"] = type(exc).__name__
                item["error"] = str(exc)

            result["candidates"].append(item)

        return result

    # =======================================================================
    # Send report
    # =======================================================================
    def _write_marker(self, report_date: str, payload: dict[str, Any]) -> None:
        # 只做最後一次寄送紀錄；不再阻止下一次寄信。
        marker = self.email_dir / f"sent_{report_date or 'unknown'}.json"
        try:
            marker.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def send_report(self, report: dict[str, Any], force: bool = False) -> dict[str, Any]:
        """
        每呼叫一次就寄一次。
        force 保留給舊版呼叫相容，不再作為去重開關。
        """
        report_date = str(report.get("report_date", ""))
        status = self.status()

        if not status["enabled"]:
            return {"sent": False, "skipped": True, "reason": "EMAIL_ENABLED=false", "status": status}

        if not status["configured"]:
            return {
                "sent": False,
                "skipped": True,
                "reason": f"Email 尚未設定：缺少 {', '.join(status['missing'])}",
                "status": status,
            }

        digest = self._agent_digest(report)
        is_morning = str(report.get("report_type", "")).lower() == "morning"
        subject = (
            f"{self.config.subject_prefix}{report_date}｜AI早報 × 法說會 × 財經新聞"
            if is_morning
            else f"{self.config.subject_prefix}{report_date}｜AI盤後重點 × 法說會 × 財報"
        )

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = formataddr((self.config.display_name, self.config.sender))
        msg["To"] = self.config.recipient
        if is_morning:
            msg.set_content(self.build_morning_text(report), charset="utf-8")
            msg.add_alternative(self.build_morning_html(report), subtype="html", charset="utf-8")
        else:
            msg.set_content(self.build_text(report), charset="utf-8")
            msg.add_alternative(self.build_html(report), subtype="html", charset="utf-8")

        attachments: list[str] = []
        if self.config.attach_reports:
            json_path = Path(str(report.get("json_path", "")))
            md_path = Path(str(report.get("markdown_path", "")))
            allowed_root = self.output_dir.resolve()

            for path, maintype, subtype in [
                (json_path, "application", "json"),
                (md_path, "text", "markdown"),
            ]:
                try:
                    rp = path.resolve()
                    if rp.exists() and allowed_root in rp.parents:
                        msg.add_attachment(
                            rp.read_bytes(),
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
                "agent_summary_status": digest.get("agent_status", "fallback"),
                "smtp_log": smtp_log,
                "status": status,
            }
            self._write_marker(report_date, result)
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

    def load_latest_morning_report(self) -> dict[str, Any] | None:
        files = sorted(self.output_dir.glob("morning_*.json"))
        if not files:
            return None
        try:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def send_latest_morning_report(self, force: bool = False) -> dict[str, Any]:
        report = self.load_latest_morning_report()
        if not report:
            return {"sent": False, "skipped": True, "reason": "目前沒有已保存的早報。", "status": self.status()}
        return self.send_report(report, force=force)

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
            return {
                "sent": False,
                "skipped": True,
                "reason": "目前沒有已保存的盤後研究報告。",
                "status": self.status(),
            }
        return self.send_report(report, force=force)


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 台股盤後財報 Email Agent v2")
    parser.add_argument("--status", action="store_true", help="顯示 Email Agent 設定")
    parser.add_argument("--send-latest", action="store_true", help="寄送最新盤後研究報告")
    parser.add_argument("--send-latest-morning", action="store_true", help="寄送最新早報")
    parser.add_argument("--force", action="store_true", help="相容舊參數；目前每次執行都會寄送")
    parser.add_argument("--diagnose-smtp", action="store_true", help="只測試 SMTP 連線，不寄信")
    args = parser.parse_args()

    agent = EmailAgent(Path(__file__).resolve().parent)

    if args.diagnose_smtp:
        print(json.dumps(agent.diagnose_smtp(), ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0)

    if args.status or (not args.send_latest and not args.send_latest_morning):
        print(json.dumps(agent.status(), ensure_ascii=False, indent=2))

    if args.send_latest_morning:
        result = agent.send_latest_morning_report(force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0 if result.get("sent") or result.get("skipped") else 1)

    if args.send_latest:
        result = agent.send_latest_report(force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0 if result.get("sent") or result.get("skipped") else 1)
