# -*- coding: utf-8 -*-
"""早報 Email：最近 N 天法說會 + 近 2 天財經新聞摘要 + AI 看好的產業與個股。

繼承 EmailAgent，重用同一套排版工具（_pct_html/_impact_badge/_co_label/_qwen_block/
_earnings_section/_cnyes_section/_base_css…）與寄信、設定讀取、防重複寄送的邏輯，
只新增早報專屬的版面組裝與寄送方法——避免兩封信共用的東西寫兩份、以後改版要改兩處。

寄送對象、主旨前綴預設沿用盤後報告同一組 EMAIL_* 設定；想寄給不同信箱／用不同主旨，
在 .env 加 MORNING_RECIPIENT / MORNING_SUBJECT_PREFIX / MORNING_AUTO_SEND 覆蓋即可，
不用整組 SMTP 帳密再填一次。
"""
from __future__ import annotations

import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any

from email_agent import EmailAgent, _as_bool, _cfg_value
from morning_report import MorningReportAgent


class MorningEmailAgent(EmailAgent):
    def _load_config(self):
        cfg = super()._load_config()

        def first(*keys: str, default: str = "") -> str:
            for key in keys:
                value = _cfg_value(self.env, key, "")
                if value:
                    return value
            return default

        cfg.recipient = first("MORNING_RECIPIENT", default=cfg.recipient)
        cfg.subject_prefix = first("MORNING_SUBJECT_PREFIX", default="【AI 台股早報】")
        cfg.display_name = first("MORNING_DISPLAY_NAME", default=cfg.display_name)
        cfg.auto_send = _as_bool(first("MORNING_AUTO_SEND", default=str(cfg.auto_send)), cfg.auto_send)
        return cfg

    def _marker_path(self, report_date: str) -> Path:
        # 跟盤後報告的寄送紀錄分開存，兩封信各自防重複寄送，互不影響。
        return self.email_dir / f"sent_morning_{report_date}.json"

    # ------------------------------- 看好產業與個股 ---------------------------------
    def _outlook_html(self, outlook: dict[str, Any], heading: str) -> str:
        outlook = outlook or {}
        if not outlook.get("available"):
            note = self._esc(outlook.get("error") or "目前沒有足夠資料可以統整看好的產業與個股。")
            return f"""
<section>
<h2>{self._esc(heading)}</h2>
<p class='muted'>{note}</p>
</section>
"""
        fallback_note = (
            "<p class='muted'>本機 Qwen3 8B 暫時無法使用，以下為資料直接排序，未經 AI 綜合判斷。</p>"
            if outlook.get("fallback") else ""
        )
        overall = outlook.get("overall_take", "")
        overall_html = f"<div class='callout-info'>{self._esc(overall)}</div>" if overall else ""

        ind_items = "".join(
            f"<li><strong>{self._esc(x.get('name',''))}</strong>：{self._esc(x.get('reason',''))}</li>"
            for x in outlook.get("industries", [])
        ) or "<li>目前沒有特別看好的產業。</li>"

        def stock_item(x: dict[str, Any]) -> str:
            label = self._co_label(x.get("name"), x.get("symbol"))
            return f"<li><strong>{self._esc(label)}</strong>：{self._esc(x.get('reason',''))}</li>"

        stock_items = "".join(stock_item(x) for x in outlook.get("stocks", [])) or "<li>目前沒有特別看好的個股。</li>"

        return f"""
<section>
<h2>{self._esc(heading)}</h2>
<p class='muted'>以下由本機 Qwen3 8B 根據近期已算好的題材強度、營收年增率、法說會正面因素統整而成，
僅為研究線索整理，不是投資建議，實際進出請自行查證。</p>
{fallback_note}
{overall_html}
<h3>看好產業／題材</h3>
<ul>{ind_items}</ul>
<h3>看好個股</h3>
<ul>{stock_items}</ul>
</section>
"""

    # ---------------------------------- 版面 ------------------------------------
    def build_morning_html(self, morning_report: dict[str, Any]) -> str:
        report_date = self._esc(morning_report.get("report_date", ""))
        lookback = morning_report.get("earnings_lookback_days", 5)
        earnings_html = self._earnings_section(
            {"earnings_calls": morning_report.get("earnings_calls", [])},
            heading=f"一、最近 {lookback} 天法說會",
            limit=30,
        )
        news_html = self._cnyes_section(morning_report.get("cnyes_news"), heading="二、近 2 天財經報導摘要")
        outlook_html = self._outlook_html(morning_report.get("outlook", {}), heading="三、AI 看好的產業與個股")

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
<h1>AI 台股早報</h1>
<div class='muted'>報告日期：{report_date}｜最近 {lookback} 天法說會＋近 2 天新聞摘要＋AI 觀察</div>
{earnings_html}
{news_html}
{outlook_html}
<div class='footer'>
本報告為盤前研究彙整與模型觀察，不保證新聞真偽，也不代表投資建議；重大消息請回看官方公告與原始新聞來源。
</div>
</div>
</body>
</html>"""

    def build_morning_text(self, morning_report: dict[str, Any]) -> str:
        lines = [
            f"AI 台股早報｜{morning_report.get('report_date','')}",
            "",
            f"【最近 {morning_report.get('earnings_lookback_days', 5)} 天法說會】",
        ]
        calls = sorted(morning_report.get("earnings_calls", []) or [],
                        key=lambda c: str(c.get("event_date", "")), reverse=True)
        for c in calls[:30]:
            lines.append(
                f"- {c.get('event_date','')} {c.get('name', c.get('symbol',''))}({c.get('symbol','')})｜"
                f"{c.get('impact','中性／待確認')}｜信心 {self._safe_float(c.get('confidence'),0):.0f}%"
            )
        if not calls:
            lines.append("- 這段期間沒有成功擷取到法說事件。")

        lines += ["", "【近 2 天財經報導摘要】"]
        news = morning_report.get("cnyes_news", {}) or {}
        digest = news.get("digest", {}) or {}
        if digest.get("available"):
            if digest.get("overall_take"):
                lines.append(f"整體重點：{digest['overall_take']}")
            for it in digest.get("items", []):
                lines.append(f"- {it.get('headline','')}（{it.get('impact','')}｜{it.get('relevance','')}）：{it.get('summary','')}")
        else:
            lines.append(f"- {digest.get('error') or '本機 Qwen3 8B 目前無法使用，暫時沒有新聞摘要。'}")

        lines += ["", "【AI 看好的產業與個股】"]
        outlook = morning_report.get("outlook", {}) or {}
        if outlook.get("available"):
            if outlook.get("fallback"):
                lines.append("（本機 Qwen3 8B 暫時無法使用，以下為資料直接排序）")
            for x in outlook.get("industries", []):
                lines.append(f"- 產業：{x.get('name','')}：{x.get('reason','')}")
            for x in outlook.get("stocks", []):
                lines.append(f"- 個股：{x.get('name', x.get('symbol',''))}({x.get('symbol','')})：{x.get('reason','')}")
        else:
            lines.append(f"- {outlook.get('error') or '目前沒有足夠資料可以統整。'}")

        lines += ["", "本報告為盤前研究彙整，不代表投資建議；完整資料請見附件或 HTML 版本。"]
        return "\n".join(lines)

    # ---------------------------------- 寄送 ------------------------------------
    def send_morning_report(self, morning_report: dict[str, Any]) -> dict[str, Any]:
        report_date = str(morning_report.get("report_date", ""))
        status = self.status()
        if not status["enabled"]:
            return {"sent": False, "skipped": True, "reason": "EMAIL_ENABLED=false", "status": status}
        if not status["configured"]:
            return {"sent": False, "skipped": True, "reason": f"Email 尚未設定：缺少 {', '.join(status['missing'])}", "status": status}

        subject = f"{self.config.subject_prefix}{report_date}｜法說會 × 財經新聞摘要 × 看好產業個股"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = formataddr((self.config.display_name, self.config.sender))
        msg["To"] = self.config.recipient
        msg.set_content(self.build_morning_text(morning_report), charset="utf-8")
        msg.add_alternative(self.build_morning_html(morning_report), subtype="html", charset="utf-8")

        try:
            smtp_log = self._smtp_send(msg)
            result = {
                "sent": True, "skipped": False, "report_date": report_date,
                "recipient": self.config.recipient, "subject": subject, "smtp_log": smtp_log, "status": status,
            }
            self._write_marker(report_date or "unknown", result)
            return result
        except smtplib.SMTPAuthenticationError as exc:
            return {"sent": False, "skipped": False, "error": f"Gmail SMTP 驗證失敗：{exc}", "status": status}
        except (smtplib.SMTPException, OSError) as exc:
            return {"sent": False, "skipped": False, "error": f"SMTP 寄信失敗：{exc}", "status": status}

    def build_and_send(self, force: bool = False) -> dict[str, Any]:
        report = MorningReportAgent(base_dir=self.base).build()
        return self.send_morning_report(report)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="AI 台股早報 Email Agent")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preview", action="store_true", help="只產生 HTML 預覽檔，不寄信")
    parser.add_argument("--send", action="store_true", help="組報告並寄出早報")
    args = parser.parse_args()

    agent = MorningEmailAgent(Path(__file__).resolve().parent)

    if args.preview:
        report = MorningReportAgent(base_dir=agent.base).build()
        preview_path = agent.email_dir / "preview_morning_latest.html"
        preview_path.write_text(agent.build_morning_html(report), encoding="utf-8")
        print(f"已產生預覽：{preview_path}")
        raise SystemExit(0)

    if args.send:
        result = agent.build_and_send()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        raise SystemExit(0 if result.get("sent") or result.get("skipped") else 1)

    print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
